#pragma once

#include "abstract_manager.hpp"
#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

// Accumulates the real-space spin-flip correlator
//
//   C(μ_i, μ_j, Δr) = ⟨S⁺_i S⁻_j⟩
//
// averaged over MC samples.  Only intra-cluster pairs contribute (classical
// and boundary spins are in Sz eigenstates so their S± is zero).
//
// At each sample, for each cluster with a precomputed eigenvector cache, the
// full thermal average is computed:
//
//   C(i,j) = (1/Z) Σ_n exp(-β E_n) ⟨n|S⁺_i S⁻_j|n⟩
//
// using the density-matrix trick  ρ = V diag(w) V^T  where V = eigenvectors.
//
// The displacement catalog is built once at construction by enumerating all
// intra-cluster spin pairs (both orderings i→j and j→i are catalogued
// separately so the FT covers both Δr and -Δr).
//
// HDF5 output group (/transverse_corr):
//   T_list         [n_T]         double   — temperatures, ascending
//   n_samples      [n_T]         uint64   — samples per temperature
//   disp_vectors   [n_disp, 3]   int64    — Δr = r_j.ipos - r_i.ipos (minimal image)
//   sublat_i       [n_disp]      int32    — sublattice index of spin i
//   sublat_j       [n_disp]      int32    — sublattice index of spin j
//   corr           [n_T, n_disp] double   — sum of per-sample ⟨S⁺_i S⁻_j⟩ (unnorm.)
//   n_pairs_per_sample [n_disp]  double   — mean pairs per sample for each disp entry
//
// Post-processing: C_mean[T,d] = corr[T,d] / (n_samples[T] * n_pairs_per_sample[d])
// FT:  S_pm(q) = Σ_d exp(i·q·Δr[d]) · C_mean[T,d] · n_pairs_per_sample[d]
class transverse_corr_manager : public abstract_manager {

    // --- displacement catalog ---
    struct DispEntry {
        ipos_t  delta;   // r_j.ipos - r_i.ipos, minimal image
        int32_t sl_i;
        int32_t sl_j;
    };

    struct DispKey {
        ipos_t  delta;
        int32_t sl_i, sl_j;
        bool operator==(const DispKey& o) const {
            return delta == o.delta && sl_i == o.sl_i && sl_j == o.sl_j;
        }
    };
    struct DispKeyHash {
        std::size_t operator()(const DispKey& k) const {
            auto h = std::hash<ipos_t>{}(k.delta);
            h ^= std::hash<int32_t>{}(k.sl_i) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.sl_j) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::vector<DispEntry>                                catalog_;
    std::unordered_map<DispKey, int, DispKeyHash>         catalog_map_;

    // n_pairs_per_sample_[d]: accumulated (sum over samples) count of pairs
    // contributing to displacement d.  Divide by n_samples to get mean.
    std::vector<double> n_pairs_per_sample_;

    // per-temperature accumulator: corr_[T_idx][disp_idx]
    std::vector<std::vector<double>> corr_;

    MCStateMF& state_;
    QClattice& sc_;
    int        n_quantum_spins_;

    void on_new_temp() override {
        corr_.emplace_back(catalog_.size(), 0.0);
    }

    // Helper: sublattice index of spin s in sc_.
    int sublattice_of(const Spin* s) const {
        const auto& all = sc_.get_objects<Spin>();
        ptrdiff_t offset = s - all.data();
        return static_cast<int>(offset / sc_.lattice.num_primitive_cells());
    }

    // Helper: minimal-image displacement (j.ipos - i.ipos), wrapped to supercell.
    ipos_t minimal_delta(const Spin* i, const Spin* j) const {
        ipos_t d = j->ipos - i->ipos;
        sc_.lattice.wrap_super_delta(d);
        return d;
    }

    // Build the displacement catalog from all intra-cluster pairs.
    void build_catalog() {
        for (const auto& cl : state_.clusters) {
            if (!cl.evec_cache) continue;
            int N = cl.n_spins();
            for (int ii = 0; ii < N; ii++) {
                for (int jj = 0; jj < N; jj++) {
                    const Spin* si = cl.spins[ii];
                    const Spin* sj = cl.spins[jj];
                    DispKey key{ minimal_delta(si, sj),
                                 static_cast<int32_t>(sublattice_of(si)),
                                 static_cast<int32_t>(sublattice_of(sj)) };
                    if (catalog_map_.count(key) == 0) {
                        catalog_map_[key] = static_cast<int>(catalog_.size());
                        catalog_.push_back({ key.delta, key.sl_i, key.sl_j });
                    }
                }
            }
        }
        n_pairs_per_sample_.assign(catalog_.size(), 0.0);
    }

public:
    // sc and state must outlive this object.
    // Call after state.clusters are initialised (i.e. after qc.initialise()).
    transverse_corr_manager(MCStateMF& state, QClattice& sc,
                            size_t n_temperatures_reserve = 0)
        : state_(state), sc_(sc), n_quantum_spins_(0)
    {
        build_catalog();
        for (const auto& cl : state_.clusters)
            n_quantum_spins_ += cl.n_spins();
        T_list.reserve(n_temperatures_reserve);
        n_samples.reserve(n_temperatures_reserve);
        corr_.reserve(n_temperatures_reserve);
    }

    // Compute thermal-average ⟨S⁺_i S⁻_j⟩ for all cached cluster pairs and accumulate.
    void sample(double beta) {
        if (T_list.empty())
            throw std::runtime_error("transverse_corr_manager: sample() before new_T()");

        auto& acc = corr_[curr_idx];

        for (const auto& cl : state_.clusters) {
            const Eigen::MatrixXd* evecs = cl.get_current_evecs();
            if (!evecs) continue;

            const int D = cl.hilbert_dim();
            const int N = cl.n_spins();

            // Boltzmann weights
            Eigen::VectorXd w(D);
            for (int n = 0; n < D; n++) w[n] = std::exp(-beta * cl.eigenvalues[n]);
            const double Z = w.sum();
            w /= Z;

            // Density matrix ρ = V diag(w) V^T  (real, symmetric, D×D)
            const Eigen::MatrixXd rho = *evecs * w.asDiagonal() * evecs->transpose();

            for (int ii = 0; ii < N; ii++) {
                for (int jj = 0; jj < N; jj++) {
                    // ⟨S⁺_ii S⁻_jj⟩ = Σ_{b: bit_jj=1, bit_ii=0} ρ(b_flip, b)
                    // where b_flip = b ^ (1<<ii) ^ (1<<jj)  (for ii != jj)
                    // For ii == jj: ⟨n̂_ii⟩ = Σ_{b: bit_ii=1} ρ(b, b)
                    double C = 0.0;
                    if (ii == jj) {
                        for (int b = 0; b < D; b++)
                            if ((b >> ii) & 1) C += rho(b, b);
                    } else {
                        for (int b = 0; b < D; b++) {
                            if (!((b >> jj) & 1)) continue; // bit_jj must be 1
                            if ( (b >> ii) & 1)  continue; // bit_ii must be 0
                            int b_flip = b ^ (1 << ii) ^ (1 << jj);
                            C += rho(b_flip, b);
                        }
                    }

                    DispKey key{ minimal_delta(cl.spins[ii], cl.spins[jj]),
                                 static_cast<int32_t>(sublattice_of(cl.spins[ii])),
                                 static_cast<int32_t>(sublattice_of(cl.spins[jj])) };
                    int d = catalog_map_.at(key);
                    acc[d] += C;
                    n_pairs_per_sample_[d] += 1.0;
                }
            }
        }

        n_samples[curr_idx]++;
    }

    void write_group(hid_t file_id, const char* group_name = "/transverse_corr") override;
};


inline void transverse_corr_manager::write_group(hid_t file_id, const char* group_name) {
    const size_t n_T    = T_list.size();
    const size_t n_disp = catalog_.size();

    // Sort by temperature ascending
    std::vector<size_t> ord(n_T);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](size_t a, size_t b_){ return T_list[a] < T_list[b_]; });

    std::vector<double>   sT(n_T);
    std::vector<uint64_t> sN(n_T);
    for (size_t i = 0; i < n_T; i++) {
        sT[i] = T_list[ord[i]];
        sN[i] = static_cast<uint64_t>(n_samples[ord[i]]);
    }

    // Normalise n_pairs_per_sample by total number of samples accumulated
    size_t total_samples = 0;
    for (size_t i = 0; i < n_T; i++) total_samples += n_samples[i];
    std::vector<double> n_pairs(n_disp);
    if (total_samples > 0) {
        for (size_t d = 0; d < n_disp; d++)
            n_pairs[d] = n_pairs_per_sample_[d] / static_cast<double>(total_samples);
    }

    hid_t grp;
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0)
        grp = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    else
        grp = H5Gcreate2(file_id, group_name,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0)
        throw std::runtime_error(
            std::string("transverse_corr_manager: failed to open/create group ") + group_name);

    auto write_1d = [&](const char* name, hid_t type, hsize_t len, const void* data) {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp);
            throw std::runtime_error(std::string("transverse_corr_manager: failed to create ") + name); }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    };

    // T_list, n_samples, n_pairs_per_sample
    write_1d("T_list",            H5T_NATIVE_DOUBLE, n_T,    sT.data());
    write_1d("n_samples",         H5T_NATIVE_UINT64, n_T,    sN.data());
    write_1d("n_pairs_per_sample",H5T_NATIVE_DOUBLE, n_disp, n_pairs.data());

    // disp_vectors [n_disp, 3]  int64
    {
        std::vector<int64_t> dv(n_disp * 3);
        for (size_t d = 0; d < n_disp; d++) {
            dv[d*3+0] = catalog_[d].delta[0];
            dv[d*3+1] = catalog_[d].delta[1];
            dv[d*3+2] = catalog_[d].delta[2];
        }
        hsize_t dims[2] = { n_disp, 3 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "disp_vectors", H5T_NATIVE_INT64, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); H5Gclose(grp);
            throw std::runtime_error("transverse_corr_manager: failed to create disp_vectors"); }
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, dv.data());
        H5Dclose(ds); H5Sclose(sp);
    }

    // sublat_i, sublat_j [n_disp]  int32
    {
        std::vector<int32_t> sli(n_disp), slj(n_disp);
        for (size_t d = 0; d < n_disp; d++) {
            sli[d] = catalog_[d].sl_i;
            slj[d] = catalog_[d].sl_j;
        }
        write_1d("sublat_i", H5T_NATIVE_INT32, n_disp, sli.data());
        write_1d("sublat_j", H5T_NATIVE_INT32, n_disp, slj.data());
    }

    // corr [n_T, n_disp]  double
    {
        hsize_t dims[2] = { n_T, n_disp };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "corr", H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); H5Gclose(grp);
            throw std::runtime_error("transverse_corr_manager: failed to create corr"); }
        std::vector<double> buf(n_T * n_disp);
        for (size_t t = 0; t < n_T; t++)
            for (size_t d = 0; d < n_disp; d++)
                buf[t * n_disp + d] = corr_[ord[t]][d];
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(ds); H5Sclose(sp);
    }

    // Scalar attribute: n_quantum_spins
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t at = H5Acreate2(grp, "n_quantum_spins", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, &n_quantum_spins_);
        H5Aclose(at); H5Sclose(sp);
    }

    H5Gclose(grp);
}

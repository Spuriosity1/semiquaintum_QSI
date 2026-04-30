#pragma once

#include "abstract_manager.hpp"
#include "quantum_cluster.hpp"  // Spin, Tetra, Plaq, QClattice
#include <fourier.hpp>          // FourierTransformC2C, FourierCorrelator, SublatWeightMatrix, correlate()
#include <vec3.hpp>             // vector3::vec3d, vector3::dot
#include "H5Apublic.h"
#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>


// Measures two complementary static structure factors for Ising spins.
//
// Accumulates the sublattice-pair correlator
//
//   C_{μν}(q) [raw] = Σ_samples conj(Ã_μ(q)) · Ã_ν(q)
//
// using lil2's FourierCorrelator.  Note: Ã_μ is the raw per-cell DFT
// output (sublattice phase exp(-i q·r_μ) excluded per lil2 convention).
// Phase factors are applied at write time via SublatWeightMatrix.
//
// Two outputs are derived at write time from the same accumulator:
//
//  Szz(q)  — local-axis <SzSz>:  Re Σ_{μν} exp(+i q·Δr_{μν}) · C_{μν}(q)
//             Pure Ising correlator; most direct view of ice-rule physics.
//
//  Sqq(q)  — physical <m·m>:     Re Σ_{μν} (ẑ_μ·ẑ_ν) exp(+i q·Δr_{μν}) · C_{μν}(q)
//             What unpolarised neutron scattering measures.
//
// Both saved in the same HDF5 group (shape n_T × n_k).
// Divide by n_spins (stored as HDF5 attribute) to get S(q) per site.
//
// Scope: correct for p=0 (no deleted spins) with no quantum clusters.
//
// Setup for pyrochlore (pass axes from pyrochlore::pyrochlore_local_axes()):
//   ssf_manager ssf(sc, pyrochlore::pyrochlore_local_axes(), n_step);
class ssf_manager : public abstract_manager {
    using IsingFT = FourierTransformC2C<
        Spin,
        FieldAccessor<Spin, &Spin::ising_val>,
        Spin, Tetra, Plaq>;

    // --- Declaration order determines initializer-list initialization order ---
    IsingFT ft;
    int n_sl;
    int n_kpoints;
    int n_spins;
    ivec3_t k_dims;
    std::vector<ipos_t> sl_positions_;

    // Precomputed weight matrices (constructed from above members)
    SublatWeightMatrix w_szz_fcc_; // not even phase factors (condenses all spins to 0,0,0)
    SublatWeightMatrix w_szz_;  // phase factors only           → Szz
    SublatWeightMatrix w_sqq_fcc_; // only axis dots (condenses all spins to 0,0,0)
    SublatWeightMatrix w_sqq_;  // phase factors * axis dots    → Sqq


    // Per-temperature accumulator of conj(Ã_μ)·Ã_ν (no phase, summed over samples)
    std::vector<FourierCorrelator<Spin>> corr_;

    void on_new_temp() override { corr_.emplace_back(n_sl, k_dims); }

    // Build n_sl × n_sl axis-dot matrix, tiling the provided axes by (sl % n_provided).
    static std::vector<std::vector<double>> make_axis_dot(
            const std::vector<vector3::vec3d>& axes, int n_sl) {
        const int np = static_cast<int>(axes.size());
        std::vector<std::vector<double>> mat(n_sl, std::vector<double>(n_sl));
        for (int mu = 0; mu < n_sl; mu++)
            for (int nu = 0; nu < n_sl; nu++)
                mat[mu][nu] = vector3::dot(axes[mu % np], axes[nu % np]);
        return mat;
    }

public:
    // sc must outlive this object.
    // local_axes: normalized quantization axes, one per physical sublattice
    // (tiled via sl % local_axes.size() to fill all n_sl lil2 sublattices).
    ssf_manager(QClattice& sc,
                const std::vector<vector3::vec3d>& local_axes,
                size_t n_temperatures_reserve = 0)
        : ft(sc),
          n_sl(static_cast<int>(std::get<SlPos<Spin>>(sc.sl_positions).size())),
          n_kpoints(sc.lattice.num_primitive_cells()),
          n_spins(static_cast<int>(sc.get_objects<Spin>().size())),
          k_dims(sc.lattice.size()),
          sl_positions_(std::get<SlPos<Spin>>(sc.sl_positions).begin(),
                        std::get<SlPos<Spin>>(sc.sl_positions).end()),
          w_szz_fcc_(SublatWeightMatrix::constant(n_sl, k_dims, 1.0)),
          w_szz_(SublatWeightMatrix::phase_factors(sc.lattice, sl_positions_)),
          w_sqq_fcc_(SublatWeightMatrix::constant(n_sl, k_dims,
                                    make_axis_dot(local_axes, n_sl))),
          w_sqq_(w_szz_ * w_sqq_fcc_)
    {
        T_list.reserve(n_temperatures_reserve);
        n_samples.reserve(n_temperatures_reserve);
        corr_.reserve(n_temperatures_reserve);
    }

    // Fourier-transform current ising_val configuration and accumulate C_{μν}(q).
    void sample() {
        assert(!T_list.empty());
        ft.transform();
//        corr_[curr_idx] += correlate(ft.get_buffer(), ft.get_buffer());
        correlate_add(corr_[curr_idx], ft.get_buffer(), ft.get_buffer());
        n_samples[curr_idx]++;
    }

    // Write Szz and Sqq (mean, not yet divided by n_spins) to the HDF5 group.
    // Physical S(q)/site = dataset / n_spins  (n_spins stored as HDF5 attribute).
    void write_group(hid_t file_id, const char* group_name = "/ssf") override;
};


inline void ssf_manager::write_group(hid_t file_id, const char* group_name) {
    const size_t n_T = T_list.size();

    std::vector<size_t> ord(n_T);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](size_t a, size_t b) { return T_list[a] < T_list[b]; });

    std::vector<double> sT(n_T);
    std::vector<size_t> sN(n_T);
    for (size_t i = 0; i < n_T; i++) {
        sT[i] = T_list[ord[i]];
        sN[i] = n_samples[ord[i]];
    }

    // Store raw sublattice correlators per temperature

    // Open or create the group
    hid_t grp;
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0)
        grp = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    else
        grp = H5Gcreate2(file_id, group_name,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0)
        throw std::runtime_error(
            std::string("ssf_manager: failed to open/create group ") + group_name);

    auto write_1d = [&](const char* name, hid_t type, hsize_t len, const void* data) {
        hid_t sp = H5Screate_simple(1, &len, nullptr);
        hid_t ds = H5Dcreate2(grp, name, type, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            throw std::runtime_error(std::string("ssf_manager: failed to create ") + name);
        }
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(sp);
    };

    // Write corr_[t](mu,nu)[k] as shape [n_T, n_sl, n_sl, n_kpoints, 2] (re/im).
    // Apply ord[] so temperature axis is sorted ascending.
    auto write_corr_raw = [&](const char* name,
            const std::vector<FourierCorrelator<Spin>>& data,
            const std::vector<size_t>& ord) {
        hsize_t ns = static_cast<hsize_t>(data[0].num_sublattices);
        hsize_t nk = static_cast<hsize_t>(n_kpoints);
        hsize_t dims[5] = { n_T, ns, ns, nk, 2 };
        hid_t sp = H5Screate_simple(5, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, name, H5T_NATIVE_DOUBLE, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) {
            H5Sclose(sp);
            H5Gclose(grp);
            throw std::runtime_error(std::string("ssf_manager: failed to create ") + name);
        }

        std::vector<double> buf(n_T * ns * ns * nk * 2);
        for (size_t t = 0; t < n_T; ++t) {
            const auto& c = data[ord[t]];
            for (hsize_t mu = 0; mu < ns; ++mu)
                for (hsize_t nu = 0; nu < ns; ++nu)
                    for (hsize_t k = 0; k < nk; ++k) {
                        const auto val = c(static_cast<int>(mu),
                                           static_cast<int>(nu))[k];
                        const size_t base = ((t * ns + mu) * ns + nu) * nk + k;
                        buf[base * 2]     = val.real();
                        buf[base * 2 + 1] = val.imag();
                    }
        }

        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(ds);
        H5Sclose(sp);
    };

    // Write sl_positions as [n_sl, 3] int64 array.
    // vec3<int64_t> has layout int64_t[3], so the vector is a flat int64 buffer.
    {
        hsize_t dims[2] = { static_cast<hsize_t>(n_sl), 3 };
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(grp, "sl_positions", H5T_NATIVE_INT64, sp,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds < 0) { H5Sclose(sp); H5Gclose(grp);
            throw std::runtime_error("ssf_manager: failed to create sl_positions"); }
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 sl_positions_.data());
        H5Dclose(ds);
        H5Sclose(sp);
    }

    write_1d("T_list",    H5T_NATIVE_DOUBLE, n_T, sT.data());
    write_1d("n_samples", H5T_NATIVE_ULLONG, n_T, sN.data());
    write_corr_raw("corr", corr_, ord);

    // Scalar attribute: n_spins
    {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t at = H5Acreate2(grp, "n_spins", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, &n_spins);
        H5Aclose(at);
        H5Sclose(sp);
    }

    // Attribute: k_dims[3]
    {
        hsize_t adim = 3;
        int kd[3] = { static_cast<int>(k_dims[0]),
                      static_cast<int>(k_dims[1]),
                      static_cast<int>(k_dims[2]) };
        hid_t sp = H5Screate_simple(1, &adim, nullptr);
        hid_t at = H5Acreate2(grp, "k_dims", H5T_NATIVE_INT, sp,
                               H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(at, H5T_NATIVE_INT, kd);
        H5Aclose(at);
        H5Sclose(sp);
    }

    H5Gclose(grp);
}

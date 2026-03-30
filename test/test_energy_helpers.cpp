// test_energy_helpers.cpp — self-consistency check for all energy helpers
// used in Monte Carlo moves.
//
// For each move type we:
//   1. Record E_before = state.energy()
//   2. Compute dE_local using exactly the formula the MC move uses
//   3. Force the state change
//   4. Record E_after = state.energy()
//   5. Assert |E_after - E_before - dE_local| < EPSILON
//
// Covered helpers:
//   A. classical_bond_energy()   — used by try_flip_classical (type-4 spins)
//   B. boundary-flip exact dE   — replicates try_flip_boundary_spin_MF_exact
//   C. cluster eigenstate dE    — replicates try_flip_cluster_state_MF
//
// Returns 0 on pass, 1 on any failure.

#include "sim_bits.hpp"
#include "monte_carlo.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// Replicate the static mf_interaction() from monte_carlo.cpp.
// Returns Jzz * sum_{bonds} <Sz_i>_n * <Sz_j>_{other eigenstate}
// (no 0.5 factor — double-counting handled by MCStateMF::energy()).
static double mf_interaction_local(const QClusterMF& qc, int n) {
    const double Jzz = ModelParams::get().Jzz;
    double E = 0.0;
    for (const auto& b : qc.mf_bonds)
        E += Jzz * qc.expect_Sz(n, b.my_site)
                 * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site);
    return E;
}

int main(int argc, char** argv) {
    double Jpm=0.1;
    if (argc > 1){
        Jpm = std::atof(argv[1]);
    }
    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = Jpm;
    ModelParams::get().Jyy = Jpm;
    const double Jzz = ModelParams::get().Jzz;

    constexpr double EPSILON = 1e-9;
    int n_total = 0, n_fail = 0;

    auto check = [&](const char* tag, double dE_pred, double dE_actual) {
        ++n_total;
        double err = std::abs(dE_actual - dE_pred);
        if (err > EPSILON) {
            ++n_fail;
            std::cerr << std::fixed << std::setprecision(12)
                      << "FAIL  " << tag
                      << "  predicted=" << dE_pred
                      << "  actual="    << dE_actual
                      << "  err="       << err << "\n";
            return false;
        }
        return true;
    };

    for (double p : {0.0, 0.05, 0.10}) {
        for (size_t trial = 0; trial < 5; ++trial) {
            const size_t seed = trial * 137 + 31;

            QClattice sc = initialise_lattice(4);
            std::mt19937 rng(seed);
            std::unordered_set<Tetra*> seed_tetras;
            MCStateMF state;

            delete_spins(rng, sc, p, seed_tetras);
            identify_1o_clusters(seed_tetras, state.clusters);
            identify_flippable_hexas(sc, state.intact_plaqs);
            for (auto& qc : state.clusters) qc.initialise();
            state.partition_spins(sc.get_objects<Spin>());

            // Randomise spin configuration
            std::uniform_int_distribution<int> coin(0, 1);
            for (auto& s : sc.get_objects<Spin>())
                if (!s.deleted) s.ising_val = coin(rng) ? +1 : -1;

            // Rebuild all cluster spectra from the current spin state.
            auto resync_clusters = [&]() {
                for (auto& qc : state.clusters) {
                    QClusterMF::BoundaryConfig cfg = 0;
                    for (int i = 0; i < (int)qc.classical_boundary_spins.size(); ++i)
                        if (qc.classical_boundary_spins[i]->ising_val == +1)
                            cfg |= (1u << i);
                    qc.diagonalise(cfg);
                }
            };
            resync_clusters();

            // ----------------------------------------------------------------
            // CHECK A: classical_bond_energy for type-4 (fully classical) spins
            //
            // Formula: dE = -2 * J * s->ising_val * sum(ising_val of classical/
            //                                            boundary neighbours)
            //        = -2 * classical_bond_energy(s, J)
            //
            // Verification: force the flip, measure actual dE from energy().
            // The quantum sector is untouched because type-4 spins have no
            // quantum neighbours (by construction), so the cluster eigenvalues
            // and MF terms are invariant.
            // ----------------------------------------------------------------
            for (Spin* s : state.classical_spins) {
                const double dE_pred = -2.0 * classical_bond_energy(s, Jzz);
                const double E_before = state.energy();
                s->ising_val *= -1;
                const double E_after = state.energy();
                s->ising_val *= -1;  // restore
                check("A: classical_bond_energy", dE_pred, E_after - E_before);
            }

            // ----------------------------------------------------------------
            // CHECK B: boundary spin flip — exact eigenvalue + MF shift
            //
            // Replicates try_flip_boundary_spin_MF_exact:
            //   dE_classical  = -2 * classical_bond_energy(s, J)
            //   dE_quantum   += (E_new[old_idx] - E_old[old_idx])
            //                 + (mf_interaction(new, old_idx) - mf_interaction(old, old_idx))
            //   for each adjacent cluster
            //
            // Verification: force flip + install new cluster spectra, measure
            // actual dE, then restore by flipping back and resyncing clusters.
            // ----------------------------------------------------------------
            for (Spin* s : state.boundary_spins) {
                struct Update {
                    QClusterMF::BoundaryConfig   new_config;
                    QClusterMF                   tmp;  // pre-computed post-flip cluster
                };
                // purely for debug
                size_t n_class_nb =0;
                size_t n_q_nb =0;


                std::map<QClusterMF*, Update> touched_clusters;

                double dE_pred = 0;

                for (Spin* nb : s->neighbours) {
                    auto* qc = static_cast<QClusterMF*>(nb->owning_cluster);
                    if (!qc) {
                        // classical spin
                        if (!nb->deleted) {
                            dE_pred -= 2.0 * Jzz * s->ising_val * nb->ising_val;
                            n_class_nb++;
                        }
                    } else {
                        // it's a quantum spin: ensure it is in dict
                        n_q_nb++;
                        touched_clusters[qc] = {qc->boundary_config, *qc};
                        // TODO resample eig index sensibly
                    }
                }

                // calculate the changes
                std::set<uint64_t> seen;

                struct MFBond {
                    QClusterMF* qc1;
                    int spin_i1;
                    QClusterMF* qc2;
                    int spin_i2;
                    double alignment(){
                        return qc1->expect_Sz(qc1->eigenstate_idx, spin_i1)*
                            qc2->expect_Sz(qc2->eigenstate_idx, spin_i2);
                    }
                };

                std::vector<MFBond> mf_bonds;

                for (auto& [qc, up] : touched_clusters){
                    up.new_config = qc->boundary_config;
                    int bidx = -1;
                    for (int i=0; i< (int)qc->classical_boundary_spins.size(); i++){
                        if (qc->classical_boundary_spins[i] == s) 
                            bidx = i;
                    }
                    assert(bidx >= 0);

                    up.new_config ^= (1u<<bidx);

                    up.tmp.diagonalise(up.new_config);
                    dE_pred += up.tmp.energy() - qc->energy();
                

                    // mf bonds correction
                    for (auto& mfb : qc->mf_bonds){
                        if (touched_clusters.contains(mfb.other)){ 
                            if ( mfb.other < qc ) continue;
                            mf_bonds.push_back({&up.tmp, mfb.my_site, 
                                        &touched_clusters[mfb.other].tmp, mfb.other_site});
                            
                        } else {
                            // other site not in update
                            mf_bonds.push_back({&up.tmp, mfb.my_site, mfb.other, mfb.other_site});
                        }
                        dE_pred -= Jzz * qc->expect_Sz(qc->eigenstate_idx, mfb.my_site) *
                            mfb.other->expect_Sz(mfb.other->eigenstate_idx, mfb.other_site);
                    }
                }

                for (auto & b : mf_bonds){
                    dE_pred += Jzz * b.alignment();
                }
                
                const double E_before = state.energy();

                // Force the state change
                s->ising_val *= -1;
                resync_clusters();
//                for (auto& [qc, up] : touched_clusters)
//                    *qc = std::move(up.tmp);

                const double E_after = state.energy();
                if (!check("B: boundary flip (exact)", dE_pred, E_after - E_before)){
                    std::cerr<< "Class. nb: "<<n_class_nb <<" Q. nb: "<<n_q_nb<<std::endl;
                }


                // Restore: flip spin back, rebuild all cluster spectra from spin state
                s->ising_val *= -1;
                resync_clusters();
            }

            // ----------------------------------------------------------------
            // CHECK C: cluster eigenstate transition (QClusterMF)
            //
            // Replicates the energy bookkeeping used by try_flip_cluster_state_MF:
            //   dE = (eigenvalues[new] - eigenvalues[old])
            //      + (mf_interaction(new) - mf_interaction(old))
            //
            // The second term matters because MCStateMF::energy() includes
            // 0.5*J*<Sz_i>*<Sz_j> cross-cluster bonds; the 0.5 appears on both
            // sides of each bond so changing one cluster's <Sz> shifts the total
            // by the full mf_interaction difference (not half).
            //
            // Verification: force the index change, measure actual dE, restore.
            // ----------------------------------------------------------------
            for (auto& qc : state.clusters) {
                const int old_idx = qc.eigenstate_idx;
                for (int new_idx = 0; new_idx < qc.hilbert_dim(); ++new_idx) {
                    if (new_idx == old_idx) continue;
                    const double dE_eval = qc.eigenvalues[new_idx] - qc.eigenvalues[old_idx];
                    const double dE_mf   = mf_interaction_local(qc, new_idx)
                                         - mf_interaction_local(qc, old_idx);
                    const double dE_pred = dE_eval + dE_mf;

                    const double E_before = state.energy();
                    qc.eigenstate_idx = new_idx;
                    const double E_after = state.energy();
                    qc.eigenstate_idx = old_idx;  // restore
                    check("C: eigenstate change", dE_pred, E_after - E_before);
                }
            }
        }
    }

    std::cout << "Total checks: " << n_total
              << "  Failures: "   << n_fail << "\n";
    return n_fail > 0 ? 1 : 0;
}

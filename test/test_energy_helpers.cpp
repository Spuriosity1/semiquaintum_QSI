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
//   D. try_flip_ring             — cluster state consistent with spin config after ring flip
//   E. try_flip_worm             — cluster state consistent with spin config after worm move
//
// Returns 0 on pass, 1 on any failure.

#include "argparse/argparse.hpp"
#include "sim_bits.hpp"
#include "monte_carlo.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// try_flip_ring is not exposed in monte_carlo.hpp.
int try_flip_ring(Plaq*, MCSettings&);

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

    auto ap = argparse::ArgumentParser("test_energy_helpers");
    ap.add_argument("Jpm")
        .help("XX coupling strength on quantum clusters")
        .default_value((double) 0.1)
        .scan<'g', double>();

    ap.add_argument("--classical")
        .help("Skip quantum cluster identification — treat all spins as classical (benchmarking)")
        .default_value(false)
        .implicit_value(true);

    ap.parse_args(argc, argv);

    bool classical_only = ap.get<bool>("--classical");

    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = ap.get<double>("Jpm");
    ModelParams::get().Jyy = ap.get<double>("Jpm");
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

    for (double p : {0.0, 0.01, 0.05, 0.10}) {
        std::cout<<"==============================\n"
            <<" p = "<<p<<"\n"
            <<"==============================\n";

        for (size_t trial = 0; trial < 5; ++trial) {
            const size_t seed = trial * 137 + 31;

            QClattice sc = initialise_lattice(4);
            std::mt19937 rng(seed);
            std::unordered_set<Tetra*> seed_tetras;
            MCStateMF state;

            delete_spins(rng, sc, p, seed_tetras);
            if (!classical_only)
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

            // Helper: save/restore full spin configuration.
            auto save_spins = [&]() {
                std::vector<int> v;
                for (auto& s : sc.get_objects<Spin>()) v.push_back(s.ising_val);
                return v;
            };
            auto restore_spins = [&](const std::vector<int>& v) {
                int k = 0;
                for (auto& s : sc.get_objects<Spin>()) s.ising_val = v[k++];
                resync_clusters();
            };

            // ----------------------------------------------------------------
            // CHECK D: try_flip_ring cluster consistency
            //
            // Forces each intact plaquette into the alternating ±1 state, then
            // calls try_flip_ring.  Verifies that state.energy() computed from
            // the cluster state written by try_flip_ring equals the energy from
            // a full resync — i.e., the cluster update is self-consistent.
            // ----------------------------------------------------------------
            {
                // beta=0 → always accept (ensures consistency check runs)
                MCSettings mc_ring;
                mc_ring.beta = 0.0;
                mc_ring.rng  = std::mt19937(seed ^ 0xCAFEBABEu);
                const auto spin_backup = save_spins();
                for (Plaq* p : state.intact_plaqs) {
                    // Force alternating ising values on the 6 ring spins.
                    for (int i = 0; i < 6; i++)
                        p->member_spins[i]->ising_val = (i % 2 == 0) ? +1 : -1;
                    resync_clusters();

                    const int flipped = try_flip_ring(p, mc_ring);
                    if (!flipped) { restore_spins(spin_backup); continue; }

                    const double E_stored = state.energy();
                    resync_clusters();
                    const double E_true = state.energy();
                    if (!check("D: ring cluster consistency", E_true, E_stored)) {
                        std::cerr << "  intact_plaqs size=" << state.intact_plaqs.size()
                                  << "  clusters=" << state.clusters.size() << "\n";
                    }
                    restore_spins(spin_backup);
                }
            }

            // ----------------------------------------------------------------
            // CHECK E: try_flip_worm cluster consistency
            //
            // Attempts a worm move from each classical spin root.  When the
            // worm closes (accepted), verifies that state.energy() from the
            // cluster state written by try_flip_worm equals the energy from a
            // full resync.
            // ----------------------------------------------------------------
            {
                MCSettings mc_worm;
                mc_worm.beta = 1.0;
                mc_worm.rng  = std::mt19937(seed ^ 0xDEADBEEFu);

                const auto spin_backup = save_spins();
                for (Spin* s : state.classical_spins) {
                    const int accepted = try_flip_worm(mc_worm, s);
                    if (!accepted) continue;

                    const double E_stored = state.energy();
                    resync_clusters();
                    const double E_true = state.energy();
                    if (!check("E: worm cluster consistency", E_true, E_stored)) {
                        std::cerr << "  classical_spins=" << state.classical_spins.size()
                                  << "  clusters=" << state.clusters.size() << "\n";
                    }
                    restore_spins(spin_backup);
                }
            }


            // ----------------------------------------------------------------
            // CHECK F: monopole worm dE self-consistency
            //
            // The worm's Metropolis step uses:
            //   dE_tetra = Jzz/2 * Σ_{intact tetras} (Q_new² - Q_old²)
            // This is exact when all path spins are purely classical (type-4).
            // When a path spin is a boundary spin (type-3, adjacent to a
            // quantum cluster), flipping it shifts the cluster eigenvalue via
            // H_boundary — a contribution absent from the tetra formula.
            //
            // This check measures dE_tetra independently and compares it to
            // the actual E_after - E_before.  A mismatch means the Metropolis
            // criterion is wrong and detailed balance is broken.
            //
            // beta=0 is used so every move is accepted, maximising coverage.
            // ----------------------------------------------------------------
            if (!state.class_tetras.empty()) {

                std::cout<<"Check F"<<std::endl;
                MCSettings mc_mon;
                mc_mon.beta = 0.0;
                mc_mon.rng  = std::mt19937(seed ^ 0xFEEDF00Du);

                size_t n_tried=0;
                size_t n_failed_F=0;

                const auto spin_backup = save_spins();

                for (Tetra* t : find_monopole_tetras(state.class_tetras)) {
                    // Save per-tetra charges and total energy before the move.
                    std::vector<int> Q_old(state.class_tetras.size());
                    for (int i = 0; i < (int)state.class_tetras.size(); i++)
                        Q_old[i] = classical_tetra_charge(state.class_tetras[i]);
                    const double E_before = state.energy();

                    const double E_class_before = state.classical_energy();
                    const double E_qc_before = state.cluster_energy();
                    const double E_mf_before = state.cluster_cluster_energy();


                    const int accepted = try_flip_monopole_worm(mc_mon, t, 4);
                    if (!accepted) continue;
                    n_tried++;

                    // Cluster-consistency sub-check: cluster state written by
                    // the worm should match a full resync.
                    const double E_stored = state.energy();
                    resync_clusters();
                    const double E_true = state.energy();
                    if (!check("F(consistency): monopole worm cluster state", E_true, E_stored))
                        std::cerr << "  clusters=" << state.clusters.size() << "\n";

                    // dE accuracy check: tetra formula vs actual energy change.
                    // Any discrepancy means Metropolis used the wrong dE.
                    const double Jzz_val = ModelParams::get().Jzz;
                    double dE_tetra = 0.0;
                    for (int i = 0; i < (int)state.class_tetras.size(); i++) {
                        int Q_new = classical_tetra_charge(state.class_tetras[i]);
                        dE_tetra += (Jzz_val / 2.0) * (Q_new*Q_new - Q_old[i]*Q_old[i]);
                    }
                    if (!check("F(dE):           monopole worm dE formula", dE_tetra, E_true - E_before)){
                        const double E_class_true = state.classical_energy();
                        const double E_qc_true = state.cluster_energy();
                        const double E_mf_true = state.cluster_cluster_energy();
                        n_failed_F++;

                        std::cerr << "  dE_tetra=" << dE_tetra
                                  << "  dE_actual=" << E_true - E_before
                                  << "\n            \tdClassical=" << E_class_true - E_class_before
                                  << "\n            \tdQuantum=" << E_qc_true - E_qc_before
                                  << "\n            \tdMF_quantum=" << E_mf_true - E_mf_before
                                  << "  diff=" << std::abs(dE_tetra - (E_true - E_before)) << "\n";
                    }
                    restore_spins(spin_backup);
                }
                std::cout<<"Check F:"<<n_tried<<" tried "<< n_failed_F<<" failed"<<std::endl;
            }
        }
    }

    std::cout << "Total checks: " << n_total
              << "  Failures: "   << n_fail << "\n";
    return n_fail > 0 ? 1 : 0;
}

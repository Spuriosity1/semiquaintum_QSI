// test_worm_step_symmetry.cpp
//
// Tests the following property of the quantum clusters:
//
//   Given any two non-quantum spins (si, sj) belonging to the same tetrahedron
//   with opposite ising_vals — exactly the situation that occurs at each step of
//   a closed worm (heal_val alternates, so the spin entering and the spin
//   leaving a tetrahedron carry opposite signs) — flipping both spins
//   simultaneously should leave the spectrum of every adjacent quantum cluster
//   such that it is possible to choose a new eigenstate index satisfying:
//
//     (a) E_new == E_old  (energy exactly preserved)
//     (b) <Sz_k>_new == <Sz_k>_old  for every cluster site k that couples to
//         an *uninvolved* boundary spin (i.e. a classical_boundary_spin of the
//         cluster that is neither si nor sj).
//
// This guarantees that the closed worm can include boundary spins without
// changing either (a) the worm's own energy or (b) the effective field that
// uninvolved boundary spins experience from the cluster, keeping inter-cluster
// mean-field couplings exact.
//
// The test covers:
//   - pairs where both si and sj are boundary spins of the *same* cluster
//     (the "trivial" case: ΔH = 0 since σ_i = −σ_j ⟹ their contributions cancel)
//   - pairs where si or sj belong to *different* clusters, or only one is
//     a boundary spin (the non-trivial case).
//
// Returns 0 on full pass, 1 if any violation is found.

#include "sim_bits.hpp"
#include "monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static const char* spin_type(const Spin* s) {
    if (s->deleted)      return "deleted";
    if (s->is_quantum()) return "quantum";
    for (Spin* nb : s->neighbours)
        if (!nb->deleted && nb->is_quantum()) return "boundary";
    return "classical";
}

// Returns cluster sites (indices into qc.spins) that are direct Ising
// neighbours of at least one uninvolved boundary spin (i.e. in
// qc.classical_boundary_spins but not equal to si or sj).
static std::vector<int> constrained_sites(const QClusterMF& qc,
                                           const Spin* si, const Spin* sj) {
    std::vector<int> out;
    int ns = (int)qc.spins.size();
    for (int k = 0; k < ns; k++) {
        const Spin* qs = qc.spins[k];
        bool coupled = false;
        for (const Spin* b : qc.classical_boundary_spins) {
            if (b == si || b == sj) continue;          // skip involved spins
            for (const Spin* nb : qs->neighbours) {
                if (nb == b) { coupled = true; break; }
            }
            if (coupled) break;
        }
        if (coupled) out.push_back(k);
    }
    return out;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = 0.1;
    ModelParams::get().Jyy = 0.1;

    constexpr double EPS = 1e-8;

    int pairs_tested    = 0;
    int energy_failures = 0;   // no eigenstate with E == E_old found after flip
    int sz_failures     = 0;   // energy match found but no such eigenstate preserves <Sz>

    for (double p : {0.05, 0.10}) {
        for (size_t trial = 0; trial < 8; trial++) {
            const size_t seed = trial * 137 + 31;

            QClattice sc = initialise_lattice(4);
            std::mt19937 rng(seed);
            std::unordered_set<Tetra*> seed_tetras;
            MCStateMF state;
            MCSettings mc;
            mc.rng.seed(seed + 999);

            delete_spins(rng, sc, p, seed_tetras);
            identify_1o_clusters(seed_tetras, state.clusters);
            identify_flippable_hexas(sc, state.intact_plaqs);
            for (auto& qc : state.clusters) qc.initialise();
            state.partition_spins(sc.get_objects<Spin>());

            // Randomise non-quantum spins to produce a varied background.
            std::uniform_int_distribution<int> coin(0, 1);
            for (auto& s : sc.get_objects<Spin>())
                if (!s.deleted && !s.is_quantum())
                    s.ising_val = coin(rng) ? +1 : -1;

            // Sync cluster boundary configs.
            for (auto& qc : state.clusters) {
                QClusterMF::BoundaryConfig cfg = 0;
                for (int i = 0; i < (int)qc.classical_boundary_spins.size(); i++)
                    if (qc.classical_boundary_spins[i]->ising_val == +1)
                        cfg |= (1u << i);
                qc.diagonalise(cfg);
            }

            // ── iterate over every tetrahedron ────────────────────────────
            for (auto& tetra : sc.get_objects<Tetra>()) {

                // Collect non-deleted, non-quantum member spins.
                std::vector<Spin*> eligible;
                for (Spin* s : tetra.member_spins)
                    if (!s->deleted && !s->is_quantum())
                        eligible.push_back(s);

                // Test every pair with opposite ising_vals that includes at
                // least one boundary spin (pure-classical pairs have no cluster
                // to perturb and are uninteresting for this test).
                for (size_t a = 0; a < eligible.size(); a++) {
                    for (size_t b = a + 1; b < eligible.size(); b++) {
                        Spin* si = eligible[a];
                        Spin* sj = eligible[b];

                        // Worm constraint: opposite ising_vals.
                        if (si->ising_val == sj->ising_val) continue;

                        const bool si_bnd = (std::string(spin_type(si)) == "boundary");
                        const bool sj_bnd = (std::string(spin_type(sj)) == "boundary");
                        if (!si_bnd && !sj_bnd) continue;  // no cluster involved

                        // Find clusters that list si or sj as a classical boundary spin.
                        std::vector<QClusterMF*> touched;
                        for (auto& qc : state.clusters) {
                            bool has_si = false, has_sj = false;
                            for (Spin* bs : qc.classical_boundary_spins) {
                                if (bs == si) has_si = true;
                                if (bs == sj) has_sj = true;
                            }
                            if (has_si || has_sj) touched.push_back(&qc);
                        }
                        if (touched.empty()) continue;

                        ++pairs_tested;

                        // ── snapshot spectra of touched clusters ──────────
                        struct Snap {
                            QClusterMF*              qc;
                            int                      old_idx;
                            double                   old_energy;
                            QClusterMF::BoundaryConfig old_bc;
                            // old_sz[n][k] = <Sz_k> in eigenstate n, old basis
                            std::vector<std::vector<double>> old_sz;
                        };
                        std::vector<Snap> snaps;
                        snaps.reserve(touched.size());
                        for (QClusterMF* qc : touched) {
                            Snap s;
                            s.qc        = qc;
                            s.old_idx   = qc->eigenstate_idx;
                            s.old_energy = qc->energy();
                            s.old_bc    = qc->boundary_config;
                            const int dim = (int)qc->eigenvalues.size();
                            const int ns  = (int)qc->spins.size();
                            s.old_sz.assign(dim, std::vector<double>(ns));
                            for (int n = 0; n < dim; n++)
                                for (int k = 0; k < ns; k++)
                                    s.old_sz[n][k] = qc->expect_Sz(n, k);
                            snaps.push_back(std::move(s));
                        }

                        // ── flip both spins and re-diagonalise ────────────
                        si->ising_val *= -1;
                        sj->ising_val *= -1;

                        for (QClusterMF* qc : touched) {
                            QClusterMF::BoundaryConfig new_cfg = 0;
                            for (int k = 0; k < (int)qc->classical_boundary_spins.size(); k++)
                                if (qc->classical_boundary_spins[k]->ising_val == +1)
                                    new_cfg |= (1u << k);
                            qc->diagonalise(new_cfg);
                        }

                        // ── check property for each touched cluster ───────
                        for (const auto& snap : snaps) {
                            QClusterMF* qc = snap.qc;
                            const int dim  = (int)qc->eigenvalues.size();

                            const std::vector<int> csites =
                                constrained_sites(*qc, si, sj);

                            bool found_energy = false;
                            bool found_full   = false;

                            for (int n = 0; n < dim; n++) {
                                if (std::abs(qc->eigenvalues[n] - snap.old_energy) > EPS)
                                    continue;
                                found_energy = true;

                                // Check <Sz> on constrained sites.
                                bool sz_ok = true;
                                for (int site : csites) {
                                    double old_sz = snap.old_sz[snap.old_idx][site];
                                    double new_sz = qc->expect_Sz(n, site);
                                    if (std::abs(new_sz - old_sz) > EPS) {
                                        sz_ok = false;
                                        break;
                                    }
                                }
                                if (sz_ok) { found_full = true; break; }
                            }

                            if (!found_energy) {
                                ++energy_failures;
                                std::cerr << "ENERGY FAIL  p=" << p
                                          << " trial=" << trial
                                          << "  E_old=" << snap.old_energy
                                          << "\n    si=" << spin_type(si)
                                          << "(σ=" << (si->ising_val) << ")"  // already flipped
                                          << "  sj=" << spin_type(sj)
                                          << "(σ=" << (sj->ising_val) << ")"
                                          << "\n    cluster size=" << qc->spins.size()
                                          << "  n_bnd=" << qc->classical_boundary_spins.size()
                                          << "  new evals:";
                                for (int n = 0; n < dim; n++)
                                    std::cerr << " " << std::fixed
                                              << std::setprecision(8)
                                              << qc->eigenvalues[n];
                                std::cerr << "\n";
                            } else if (!found_full) {
                                ++sz_failures;
                                std::cerr << "SZ FAIL  p=" << p
                                          << " trial=" << trial
                                          << "  E_old=" << snap.old_energy
                                          << "\n    si=" << spin_type(si)
                                          << "  sj=" << spin_type(sj)
                                          << "  cluster size=" << qc->spins.size()
                                          << "  constrained_sites=" << csites.size()
                                          << "\n    old <Sz> at constrained sites (eigenstate "
                                          << snap.old_idx << "):";
                                for (int site : csites)
                                    std::cerr << " " << std::fixed
                                              << std::setprecision(8)
                                              << snap.old_sz[snap.old_idx][site];
                                std::cerr << "\n";
                            }
                        }

                        // ── restore: unflip spins, re-diagonalise old cfg ─
                        si->ising_val *= -1;
                        sj->ising_val *= -1;
                        for (const auto& snap : snaps) {
                            snap.qc->diagonalise(snap.old_bc);   // restores exact spectrum from cache
                            snap.qc->eigenstate_idx = snap.old_idx;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Pairs_tested="    << pairs_tested
              << "  Energy_failures=" << energy_failures
              << "  Sz_failures="     << sz_failures
              << "\n";

    return (energy_failures + sz_failures) > 0 ? 1 : 0;
}

// test_worm_step_symmetry.cpp
//
// Tests the following symmetry property of quantum clusters adjacent to a
// closed-worm (hexagonal ring) update:
//
//   When all 6 spins of a pyrochlore hexagonal plaquette are simultaneously
//   flipped (the ring-worm move), every quantum cluster that has one or more
//   of those spins as a classical boundary spin must admit a new eigenstate
//   index such that
//
//     (a) its energy exactly equals the cluster's energy before the flip, and
//     (b) <Sz_k>_new == <Sz_k>_old for every cluster site k that couples
//         to an *uninvolved* boundary spin (a classical_boundary_spin not in
//         the hexagonal ring).
//
// Physical motivation: the alternating-spin pattern of a flippable hexagon
// means that each cluster's ring-boundary spins appear in consecutive pairs
// with opposite ising_vals.  Their contributions to H_boundary cancel
// (ΔH = 0), so the Hamiltonian is unchanged and the old eigenstate trivially
// satisfies both conditions.  The test verifies this holds in the actual code
// with exact arithmetic.
//
// Only plaquettes whose all 6 members are non-deleted and non-quantum
// (is_complete == true) and that are in the strict alternating state
// (the actual closed-worm precondition) are tested.
//
// Returns 0 on full pass, 1 if any violation is found.

#include "sim_bits.hpp"
#include "monte_carlo.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static const char* spin_type(const Spin* s) {
    if (s->deleted)      return "deleted";
    if (s->is_quantum()) return "quantum";
    for (const Spin* nb : s->neighbours)
        if (!nb->deleted && nb->is_quantum()) return "boundary";
    return "classical";
}

// Returns true if plaquette is in the strict alternating state required for
// a closed worm: every consecutive pair of member spins has opposite ising_val.
static bool is_alternating(const Plaq& p) {
    int prev = p.member_spins[0]->ising_val;
    for (int i = 1; i < 6; i++) {
        int cur = p.member_spins[i]->ising_val;
        if (cur * prev != -1) return false;
        prev = cur;
    }
    return true;
}

// Cluster site indices (into qc.spins) whose Ising coupling to the
// cluster Hamiltonian involves at least one *uninvolved* boundary spin
// (i.e. a classical_boundary_spin of qc that is NOT one of the ring spins).
static std::vector<int> constrained_sites(
        const QClusterMF& qc,
        const std::array<Spin*, 6>& ring)
{
    std::vector<int> out;
    for (int k = 0; k < (int)qc.spins.size(); k++) {
        const Spin* qs = qc.spins[k];
        bool coupled_to_uninvolved = false;
        for (const Spin* b : qc.classical_boundary_spins) {
            // Is b a ring spin?  Skip if so.
            bool in_ring = false;
            for (const Spin* rs : ring) if (b == rs) { in_ring = true; break; }
            if (in_ring) continue;
            // Is qs a direct neighbour of b?
            for (const Spin* nb : qs->neighbours) {
                if (nb == b) { coupled_to_uninvolved = true; break; }
            }
            if (coupled_to_uninvolved) break;
        }
        if (coupled_to_uninvolved) out.push_back(k);
    }
    return out;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = 0.1;
    ModelParams::get().Jyy = 0.1;

    constexpr double EPS = 1e-8;

    int rings_tested    = 0;
    int energy_failures = 0;   // (a) violated: no new eigenstate has E == E_old
    int sz_failures     = 0;   // (b) violated: energy match exists but no such
                               //     eigenstate preserves <Sz> on constrained sites

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

            // Randomise non-quantum spins.
            std::uniform_int_distribution<int> coin(0, 1);
            for (auto& s : sc.get_objects<Spin>())
                if (!s.deleted && !s.is_quantum())
                    s.ising_val = coin(rng) ? +1 : -1;

            // Sync cluster boundary configs after randomisation.
            for (auto& qc : state.clusters) {
                QClusterMF::BoundaryConfig cfg = 0;
                for (int i = 0; i < (int)qc.classical_boundary_spins.size(); i++)
                    if (qc.classical_boundary_spins[i]->ising_val == +1)
                        cfg |= (1u << i);
                qc.diagonalise(cfg);
            }

            // ── iterate over flippable hexagonal plaquettes ───────────────
            for (Plaq* p : state.intact_plaqs) {
                // Only test rings that are in the strict alternating state
                // (the actual precondition for a closed worm move).
                if (!is_alternating(*p)) continue;

                const auto& ring = p->member_spins;

                // Find clusters that have at least one ring spin as a
                // classical boundary spin.
                std::vector<QClusterMF*> touched;
                for (auto& qc : state.clusters) {
                    bool found = false;
                    for (const Spin* rs : ring) {
                        for (const Spin* bs : qc.classical_boundary_spins) {
                            if (bs == rs) { found = true; break; }
                        }
                        if (found) break;
                    }
                    if (found) touched.push_back(&qc);
                }
                if (touched.empty()) continue;

                ++rings_tested;

                // ── snapshot every touched cluster ────────────────────────
                struct Snap {
                    QClusterMF*                qc;
                    int                        old_idx;
                    double                     old_energy;
                    QClusterMF::BoundaryConfig old_bc;
                    // old_sz[n][k] = <Sz_k> in eigenstate n, before flip
                    std::vector<std::vector<double>> old_sz;
                };
                std::vector<Snap> snaps;
                snaps.reserve(touched.size());
                for (QClusterMF* qc : touched) {
                    Snap s;
                    s.qc         = qc;
                    s.old_idx    = qc->eigenstate_idx;
                    s.old_energy = qc->energy();
                    s.old_bc     = qc->boundary_config;
                    const int dim = (int)qc->eigenvalues.size();
                    const int ns  = (int)qc->spins.size();
                    s.old_sz.assign(dim, std::vector<double>(ns));
                    for (int n = 0; n < dim; n++)
                        for (int k = 0; k < ns; k++)
                            s.old_sz[n][k] = qc->expect_Sz(n, k);
                    snaps.push_back(std::move(s));
                }

                // ── flip all 6 ring spins ─────────────────────────────────
                for (Spin* s : ring) s->ising_val *= -1;

                // ── re-diagonalise touched clusters ───────────────────────
                for (QClusterMF* qc : touched) {
                    QClusterMF::BoundaryConfig new_cfg = 0;
                    for (int k = 0; k < (int)qc->classical_boundary_spins.size(); k++)
                        if (qc->classical_boundary_spins[k]->ising_val == +1)
                            new_cfg |= (1u << k);
                    qc->diagonalise(new_cfg);
                }

                // ── verify property for each touched cluster ──────────────
                for (const auto& snap : snaps) {
                    QClusterMF* qc  = snap.qc;
                    const int   dim = (int)qc->eigenvalues.size();

                    const std::vector<int> csites = constrained_sites(*qc, ring);

                    bool found_energy = false;
                    bool found_full   = false;

                    for (int n = 0; n < dim; n++) {
                        // (a) energy must match exactly.
                        if (std::abs(qc->eigenvalues[n] - snap.old_energy) > EPS)
                            continue;
                        found_energy = true;

                        // (b) <Sz_k> must match for every constrained site.
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
                                  << "  E_old=" << std::fixed
                                  << std::setprecision(10) << snap.old_energy
                                  << "  cluster_size=" << qc->spins.size()
                                  << "  n_ring_bnd=";
                        int nrb = 0;
                        for (const Spin* bs : qc->classical_boundary_spins)
                            for (const Spin* rs : ring) if (bs == rs) nrb++;
                        std::cerr << nrb
                                  << "\n  ring spin types:";
                        for (const Spin* rs : ring)
                            std::cerr << " " << spin_type(rs);
                        std::cerr << "\n  new eigenvalues:";
                        for (int n = 0; n < dim; n++)
                            std::cerr << " " << qc->eigenvalues[n];
                        std::cerr << "\n";

                    } else if (!found_full) {
                        ++sz_failures;
                        std::cerr << "SZ FAIL  p=" << p
                                  << " trial=" << trial
                                  << "  E_old=" << snap.old_energy
                                  << "  cluster_size=" << qc->spins.size()
                                  << "  constrained_sites=" << csites.size()
                                  << "\n  old <Sz> at constrained sites"
                                  << " (eigenstate " << snap.old_idx << "):";
                        for (int site : csites)
                            std::cerr << " " << std::fixed << std::setprecision(8)
                                      << snap.old_sz[snap.old_idx][site];
                        std::cerr << "\n";
                    }
                }

                // ── restore: unflip ring, re-diagonalise with old bc ──────
                for (Spin* s : ring) s->ising_val *= -1;
                for (const auto& snap : snaps) {
                    snap.qc->diagonalise(snap.old_bc);   // exact restore from cache
                    snap.qc->eigenstate_idx = snap.old_idx;
                }
            }
        }
    }

    std::cout << "Rings_tested="    << rings_tested
              << "  Energy_failures=" << energy_failures
              << "  Sz_failures="     << sz_failures
              << "\n";

    return (energy_failures + sz_failures) > 0 ? 1 : 0;
}

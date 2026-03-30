// test_monopole_worm.cpp
//
// Verifies that try_flip_monopole_worm's ΔE formula
//
//   dE_formula = Jzz/2 · Σ_{tetras where Q changed} (Q_f² − Q_i²)
//
// matches the true full-system energy change dE_true = E_after − E_before
// as computed by MCStateMF::energy() (the sigma-convention reference).
//
// Separately checks that at most 2 tetrahedra change their charge per worm
// (the head and tail; all intermediate tetras must cancel).
//
// On any failure the classical / boundary / quantum makeup of every affected
// tetrahedron is printed so the source of the error can be pinpointed.
//
// Returns 0 on pass, 1 if any violation found.

#include "sim_bits.hpp"
#include "monte_carlo.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

static int classical_tetra_charge(const Tetra* t) {
    int q = 0;
    for (Spin* s : t->member_spins) {
        if (s->deleted) continue;
        assert(!s->is_quantum());
        q += s->ising_val;
    }
    return q;
}

static const char* spin_type(const Spin* s) {
    if (s->deleted)      return "deleted";
    if (s->is_quantum()) return "quantum";
    for (Spin* nb : s->neighbours)
        if (!nb->deleted && nb->is_quantum()) return "boundary";
    return "classical";
}

// One-line description of a tetrahedron: lists every member spin's type and
// current σ value, and whether the tetra is in intact_tetras.
static std::string tetra_desc(const Tetra* t,
                               const std::vector<Tetra*>& intact_tetras,
                               int Q_before, int Q_after) {
    bool is_intact = false;
    for (const Tetra* it : intact_tetras) if (it == t) { is_intact = true; break; }

    std::string s = std::string(is_intact ? "[intact] " : "[non-intact] ");
    s += "Q: " + std::to_string(Q_before) + " → " + std::to_string(Q_after) + "  members:";
    for (Spin* m : t->member_spins) {
        s += "  ";
        s += spin_type(m);
        s += "(σ=";
        s += std::to_string(m->ising_val);
        s += ")";
    }
    return s;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = 0.1;
    ModelParams::get().Jyy = 0.1;

    constexpr double EPSILON = 1e-9;
    int total_accepted   = 0;
    int formula_failures = 0;   // |dE_formula − dE_true| > EPSILON
    int count_failures   = 0;   // more than 2 tetras changed charge

    // p=0  : pure classical — formula must hold exactly.
    // p>0  : quantum clusters present — quantum energy changes show up as
    //         formula_failures, letting us gauge the adiabatic-following error.
    for (double p : {0.0, 0.05, 0.10}) {
        for (size_t trial = 0; trial < 8; trial++) {
            const size_t seed = trial * 137 + 31;

            QClattice sc = initialise_lattice(4);
            std::mt19937 rng(seed);
            std::unordered_set<Tetra*> seed_tetras;
            MCStateMF state;
            MCSettings mc;
            mc.beta = 0.0;       // beta = 0 ⟹ exp(−β·dE) = 1 ⟹ every worm accepted
            mc.rng.seed(seed + 999);

            delete_spins(rng, sc, p, seed_tetras);
            identify_1o_clusters(seed_tetras, state.clusters);
            identify_flippable_hexas(sc, state.intact_plaqs);
            for (auto& qc : state.clusters) qc.initialise();
            state.partition_spins(sc.get_objects<Spin>());

            // Randomise spins to create a rich monopole background.
            std::uniform_int_distribution<int> coin(0, 1);
            for (auto& s : sc.get_objects<Spin>())
                if (!s.deleted && !s.is_quantum()) s.ising_val = coin(rng) ? +1 : -1;

            // Sync cluster boundary configs after randomisation.
            for (auto& qc : state.clusters) {
                QClusterMF::BoundaryConfig cfg = 0;
                for (int i = 0; i < (int)qc.classical_boundary_spins.size(); i++)
                    if (qc.classical_boundary_spins[i]->ising_val == +1)
                        cfg |= (1u << i);
                qc.diagonalise(cfg);
            }

            auto& all_tetras = sc.get_objects<Tetra>();
            const double Jzz = ModelParams::get().Jzz;

            for (Tetra* t : state.intact_tetras) {
                if (classical_tetra_charge(t) == 0) continue;

                // Snapshot charges of every tetrahedron (intact and non-intact).
                std::vector<int> Q_before(all_tetras.size());
                for (size_t i = 0; i < all_tetras.size(); i++)
                    Q_before[i] = classical_tetra_charge(&all_tetras[i]);

                const double E_before = state.energy();

                // Use a moderate mean worm length to exercise intermediate tetras.
                if (!try_flip_monopole_worm(mc, t, 12.0)) continue;
                total_accepted++;

                const double E_after = state.energy();
                const double dE_true = E_after - E_before;

                // Collect all tetras whose charge changed and compute formula dE.
                double dE_formula = 0.0;
                int    n_changed  = 0;
                struct ChangedTetra { size_t idx; int Q_i; int Q_f; };
                std::vector<ChangedTetra> changed;

                for (size_t i = 0; i < all_tetras.size(); i++) {
                    int Qf = classical_tetra_charge(&all_tetras[i]);
                    if (Qf != Q_before[i]) {
                        dE_formula += (Jzz / 2.0) *
                            (double)(Qf * Qf - Q_before[i] * Q_before[i]);
                        changed.push_back({i, Q_before[i], Qf});
                        ++n_changed;
                    }
                }

                // ── Check 1: formula matches full energy ──────────────────
                if (std::abs(dE_true - dE_formula) > EPSILON) {
                    ++formula_failures;
                    std::cerr << std::fixed << std::setprecision(10)
                              << "FORMULA FAIL  p=" << p
                              << " trial=" << trial
                              << "  dE_true="    << dE_true
                              << "  dE_formula=" << dE_formula
                              << "  diff="       << (dE_true - dE_formula)
                              << "\n  Changed tetrahedra (" << n_changed << "):\n";
                    for (auto& c : changed) {
                        std::cerr << "    "
                                  << tetra_desc(&all_tetras[c.idx],
                                                state.intact_tetras,
                                                c.Q_i, c.Q_f)
                                  << "\n";
                    }
                }

                // ── Check 2: at most head + tail changed charge ───────────
                if (n_changed > 2) {
                    ++count_failures;
                    std::cerr << "COUNT FAIL  p=" << p
                              << " trial=" << trial
                              << "  n_changed=" << n_changed
                              << " (expected ≤ 2)\n"
                              << "  Changed tetrahedra:\n";
                    for (auto& c : changed) {
                        std::cerr << "    "
                                  << tetra_desc(&all_tetras[c.idx],
                                                state.intact_tetras,
                                                c.Q_i, c.Q_f)
                                  << "\n";
                    }
                }
            }
        }
    }

    std::cout << "Accepted=" << total_accepted
              << "  Formula_failures=" << formula_failures
              << "  Count_failures="   << count_failures
              << "\n";

    return (formula_failures + count_failures) > 0 ? 1 : 0;
}

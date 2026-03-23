// test_worm.cpp — verify that try_flip_worm never changes the total system energy.
//
// For each accepted worm attempt, asserts |E_after − E_before| < 1e-9.
// Tests both p=0 (pure classical) and p>0 (with quantum clusters and boundary spins).
// Spin configurations are randomised so we are not restricted to the ice manifold.
//
// Returns 0 on pass, 1 if any violation is found.

#include "sim_bits.hpp"
#include "monte_carlo.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    ModelParams::get().Jzz = 1.0;
    ModelParams::get().Jxx = 0.1;
    ModelParams::get().Jyy = 0.1;

    constexpr double EPSILON = 1e-9;

    int total_attempted  = 0;
    int total_accepted   = 0;
    int total_violations = 0;

    for (double p : {0.0, 0.05, 0.10}) {
        for (size_t trial = 0; trial < 5; trial++) {
            const size_t seed = trial * 137 + 31;

            QClattice sc = initialise_lattice(4);
            std::mt19937 rng(seed);
            std::unordered_set<Tetra*> seed_tetras;
            MCStateMF state;
            MCSettings mc;
            mc.beta = 1.0;
            mc.rng.seed(seed + 999);

            delete_spins(rng, sc, p, seed_tetras);
            identify_1o_clusters(seed_tetras, state.clusters);
            identify_flippable_hexas(sc, state.intact_plaqs);
            for (auto& qc : state.clusters) qc.initialise();
            state.partition_spins(sc.get_objects<Spin>());

            // Randomise spin configuration and re-sync cluster boundary configs.
            std::uniform_int_distribution<int> coin(0, 1);
            for (auto& s : sc.get_objects<Spin>())
                if (!s.deleted) s.ising_val = coin(rng) ? +1 : -1;
            for (auto& qc : state.clusters) {
                QClusterMF::BoundaryConfig cfg = 0;
                for (int i = 0; i < (int)qc.classical_boundary_spins.size(); i++)
                    if (qc.classical_boundary_spins[i]->ising_val == +1)
                        cfg |= (1u << i);
                qc.diagonalise(cfg);
            }

            // Try a worm from every non-deleted, non-quantum spin.
            for (auto& spin : sc.get_objects<Spin>()) {
                if (spin.deleted || spin.is_quantum()) continue;

                const double E_before = state.energy();
                const int accepted = try_flip_worm(mc, &spin);
                const double E_after = state.energy();

                total_attempted++;
                if (accepted) {
                    total_accepted++;
                    const double dE = std::abs(E_after - E_before);
                    if (dE > EPSILON) {
                        total_violations++;
                        std::cerr << std::fixed << std::setprecision(12)
                                  << "VIOLATION  p=" << p
                                  << "  trial=" << trial
                                  << "  dE=" << dE
                                  << "\n";
                    }
                }
            }
        }
    }

    std::cout << "Attempted=" << total_attempted
              << "  Accepted=" << total_accepted
              << "  Violations=" << total_violations
              << "\n";

    return total_violations > 0 ? 1 : 0;
}

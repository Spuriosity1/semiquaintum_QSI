// sq_pyrochlore_compare.cpp
//
// Runs two independent MC chains on identical lattice realisations — one using
// try_flip_boundary_spin_MF (⟨Sz⟩ mean-field, no speculative re-diag) and one
// using try_flip_boundary_spin_MF_exact (full eigenvalue shift) — and emits a
// CSV comparing observables at each temperature.
//
// CSV columns:
//   T, E_mf, E_exact, var_mf, var_exact,
//   acc_boundary_mf, acc_boundary_exact,
//   ms_per_sweep_mf, ms_per_sweep_exact

#include "monte_carlo.hpp"
#include "quantum_cluster.hpp"
#include "sim_bits.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <unordered_set>

// Accumulate E and E² over nsweep sweeps; return {mean/N, variance/N, ms_per_sweep}.
template<bool UseExact>
static std::tuple<double, double, double>
run_temperature(MCStateMF& state, MCSettings& mc, size_t nsweep, int N) {
    double E_sum = 0, E2_sum = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (size_t n = 0; n < nsweep; n++) {
        state.sweep<UseExact>(mc);
        double e = state.energy();
        E_sum  += e;
        E2_sum += e * e;
    }
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() / nsweep;

    double mean = E_sum  / nsweep;
    double var  = E2_sum / nsweep - mean * mean;
    return {mean / N, var / N, ms};
}

int main(int argc, char* argv[]) {
    auto ap = argparse::ArgumentParser("sq_pyrochlore_compare");

    ap.add_argument("L").scan<'i', int>();
    ap.add_argument("p").scan<'g', float>();
    ap.add_argument("--Jzz").default_value(1.0).scan<'g', double>();
    ap.add_argument("--Jxx").default_value(0.1).scan<'g', double>();
    ap.add_argument("--Jyy").default_value(0.1).scan<'g', double>();
    ap.add_argument("--seed", "-s")
        .default_value(static_cast<size_t>(0)).scan<'i', size_t>();
    ap.add_argument("--nburn")
        .default_value(static_cast<size_t>(100)).scan<'i', size_t>();
    ap.add_argument("--nsweep", "-w").scan<'i', size_t>();
    ap.add_argument("--nstep")
        .default_value(static_cast<size_t>(20)).scan<'i', size_t>();
    ap.add_argument("--Thot").default_value(1.0).scan<'g', double>();
    ap.add_argument("--Tcold").default_value(0.001).scan<'g', double>();

    ap.parse_args(argc, argv);

    const int    L      = ap.get<int>("L");
    const double p      = ap.get<float>("p");
    const size_t seed   = ap.get<size_t>("--seed");
    const size_t nsweep = ap.get<size_t>("--nsweep");
    const size_t nburn  = ap.get<size_t>("--nburn");
    const size_t nstep  = ap.get<size_t>("--nstep");
    const double Thot   = ap.get<double>("--Thot");
    const double Tcold  = ap.get<double>("--Tcold");
    const int    N      = 16 * L * L * L;

    ModelParams::get().Jzz = ap.get<double>("--Jzz");
    ModelParams::get().Jxx = ap.get<double>("--Jxx");
    ModelParams::get().Jyy = ap.get<double>("--Jyy");

    // Build two identical lattice realisations.
    // Both QClattice objects must outlive their corresponding MCStateMF, because
    // MCStateMF holds raw Spin* / Plaq* pointers into the lattice's storage.
    auto sc_mf    = initialise_lattice(L);
    auto sc_exact = initialise_lattice(L);

    MCStateMF state_mf, state_exact;

    auto init_state = [&](QClattice& sc, MCStateMF& state) {
        std::mt19937 rng(seed);
        std::unordered_set<Tetra*> seed_tetras;
        delete_spins(rng, sc, p, seed_tetras);
        identify_1o_clusters(seed_tetras, state.clusters);
        identify_flippable_hexas(sc, state.intact_plaqs);
        for (auto& qc : state.clusters) qc.initialise();
        state.partition_spins(sc.get_objects<Spin>());
    };

    init_state(sc_mf,    state_mf);
    init_state(sc_exact, state_exact);

    output_cluster_dist(std::cerr, state_mf.clusters, 1);
    std::cerr << state_mf.classical_spins.size() << " classical | "
              << state_mf.boundary_spins.size()  << " boundary | "
              << state_mf.clusters.size()         << " clusters\n";

    MCSettings mc_mf, mc_exact;
    mc_mf.rng.seed(seed + 1);
    mc_exact.rng.seed(seed + 1);
    mc_mf.beta = mc_exact.beta = 1.0 / Thot;

    std::cerr << "Burning in (" << nburn << " sweeps)..." << std::flush;
    for (size_t n = 0; n < nburn; n++) {
        state_mf.sweep<false>(mc_mf);
        state_exact.sweep<true>(mc_exact);
    }
    std::cerr << " done.\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "T"
              << ",E_mf,E_exact"
              << ",var_mf,var_exact"
              << ",acc_boundary_mf,acc_boundary_exact"
              << ",ms_per_sweep_mf,ms_per_sweep_exact"
              << "\n";

    const double factor = std::exp(std::log(Tcold / Thot) / nstep);

    for (size_t i = 0; i < nstep; i++) {
        const double T = 1.0 / mc_mf.beta;
        mc_mf.reset_acceptance();
        mc_exact.reset_acceptance();

        auto [Em,    vm,    msm]  = run_temperature<false>(state_mf,    mc_mf,    nsweep, N);
        auto [Ee,    ve,    mse]  = run_temperature<true> (state_exact, mc_exact, nsweep, N);

        const size_t nb = state_mf.boundary_spins.size();
        double acc_b_mf    = nb ? mc_mf.accepted_boundary    / (mc_mf.sweeps_attempted    * nb) : 0.0;
        double acc_b_exact = nb ? mc_exact.accepted_boundary / (mc_exact.sweeps_attempted * nb) : 0.0;

        std::cout << T
                  << "," << Em  << "," << Ee
                  << "," << vm  << "," << ve
                  << "," << acc_b_mf << "," << acc_b_exact
                  << "," << msm << "," << mse
                  << "\n";
        std::cout.flush();

        mc_mf.beta    /= factor;
        mc_exact.beta /= factor;
    }

    return 0;
}

#pragma once

#include "quantum_cluster.hpp"
#include <random>
#include <set>
#include <sstream>

// Ising ZZ bond energy of spin s with all of its non-deleted, non-quantum neighbours.
// Used to compute ΔE = −2 * classical_bond_energy(s) when s is flipped.
// Does NOT include bonds to quantum spins (those are inside the cluster eigenvalue).
inline double classical_bond_energy(const Spin* s, double J) {
    double E = 0;
    for (const Spin* nb : s->neighbours) {
        if (!nb->deleted && !nb->is_quantum())
            E += J * s->ising_val * nb->ising_val / 4;
    }
    return E;
}

// ---- MC moves ----

struct MCSettings {
    double beta;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    double accepted_classical=0;
    double accepted_boundary=0;
    double accepted_quantum=0;
    double accepted_plaq=0;
    double accepted_worm=0;

    size_t sweeps_attempted=0;

    void reset_acceptance(){
        accepted_classical=0;
        accepted_boundary=0;
        accepted_quantum=0;
        accepted_worm=0;
        sweeps_attempted=0;
    }

    std::string acceptance(){
        std::ostringstream os;
        os << "Acc R="<<100.0 * accepted_plaq/sweeps_attempted<<"%\t"
         << "Acc W="<< 100.0 * accepted_worm/sweeps_attempted<<"%\t"
         << "Acc C="<< 100.0 * accepted_classical/sweeps_attempted<<"%\t"
         << "Acc B="<< 100.0 * accepted_boundary/sweeps_attempted<<"%\t"
         << "Acc Q="<< 100.0 * accepted_quantum/sweeps_attempted<<"%";
        return os.str();
    }

};



struct MCState {
    std::vector<Spin*>    classical_spins;  // type 4
    std::vector<Spin*>    boundary_spins;  // type 3
    std::vector<QCluster> clusters;

    void partition_spins(std::vector<Spin>& spins);

    void sweep(MCSettings& mc_);

    double energy();
};



// Forward declarations for free MC-move functions defined in monte_carlo.cpp.
int try_flip_boundary_spin_MF_exact(MCSettings& mc, Spin* s);
int try_flip_worm(MCSettings& mc, Spin* root);

struct MCStateMF {
    std::vector<Plaq*>    intact_plaqs;
    std::vector<Spin*>    classical_spins;  // type 4
    std::vector<Spin*>    boundary_spins;  // type 3
    std::vector<QClusterMF> clusters;

    void partition_spins(std::vector<Spin>& spins);

    // UseExactBoundary=false: MF ⟨Sz⟩ estimate, no speculative re-diagonalisation.
    // UseExactBoundary=true:  exact eigenvalue shift via speculative re-diagonalisation.
    // if constexpr eliminates the branch at compile time.
    template<bool UseExactBoundary = false>
    void sweep(MCSettings& mc_);

    double energy();
};

extern template void MCStateMF::sweep<false>(MCSettings&);
extern template void MCStateMF::sweep<true>(MCSettings&);

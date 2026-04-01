#pragma once

#include "quantum_cluster.hpp"
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

// Ising ZZ bond energy of spin s with all of its non-deleted, non-quantum neighbours.
// Used to compute ΔE = −2 * classical_bond_energy(s) when s is flipped.
// Does NOT include bonds to quantum spins (those are inside the cluster eigenvalue).
inline double classical_bond_energy(const Spin* s, double J) {
    double E = 0;
    for (const Spin* nb : s->neighbours) {
        if (!nb->deleted && !nb->is_quantum())
            E += J * s->ising_val * nb->ising_val;
    }
    return E;
}

// ---- MC moves ----

enum MoveFlags : uint32_t {
    MOVE_RING      = 1u << 0,  // R: flippable-hexagon ring
    MOVE_CLASSICAL = 1u << 1,  // C: single-spin Metropolis
    MOVE_WORM      = 1u << 2,  // W: closed-loop worm
    MOVE_MONOPOLE  = 1u << 3,  // M: monopole worm
    MOVE_BOUNDARY  = 1u << 4,  // B: boundary-spin flip
    MOVE_QUANTUM   = 1u << 5,  // Q: cluster eigenstate resample
    MOVE_ALL       = ~0u,
    MOVE_HIGH_T    = MOVE_CLASSICAL | MOVE_QUANTUM | MOVE_BOUNDARY
};

inline MoveFlags parse_moves(const std::string& s) {
    uint32_t f = 0;
    for (char c : s) {
        switch (c) {
            case 'R': case 'r': f |= MOVE_RING;      break;
            case 'C': case 'c': f |= MOVE_CLASSICAL; break;
            case 'W': case 'w': f |= MOVE_WORM;      break;
            case 'M': case 'm': f |= MOVE_MONOPOLE;  break;
            case 'B': case 'b': f |= MOVE_BOUNDARY;  break;
            case 'Q': case 'q': f |= MOVE_QUANTUM;   break;
            default: throw std::invalid_argument(
                std::string("Unknown move letter '") + c + "' (valid: RCWMBQ)");
        }
    }
    return static_cast<MoveFlags>(f);
}

struct MCSettings {
    double beta;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    MoveFlags moves = MOVE_ALL;

    double accepted_classical=0;
    double accepted_boundary=0;
    double accepted_quantum=0;
    double accepted_plaq=0;
    double accepted_worm=0;
    double accepted_monopole=0;

    size_t attempted_monopole=0;
    size_t sweeps_attempted=0;

    void reset_acceptance(){
        accepted_classical=0;
        accepted_boundary=0;
        accepted_quantum=0;
        accepted_worm=0;
        accepted_monopole=0;
        attempted_monopole=0;
        sweeps_attempted=0;
    }

    std::string acceptance(){
        std::ostringstream os;
        os << "Acc R="<<100.0 * accepted_plaq/sweeps_attempted<<"%\t"
         << "Acc W="<< 100.0 * accepted_worm/sweeps_attempted<<"%\t"
         << "Acc M="<< (attempted_monopole > 0
                        ? std::to_string(100.0 * accepted_monopole/attempted_monopole) + "%"
                        : std::string("n/a"))<<"\t"
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

// utility funciton (used by a test, otherwise could be private)

// filters out a list of tetras with monopoles on them
int classical_tetra_charge(const Tetra* t);
std::vector<Tetra*> find_monopole_tetras(const std::vector<Tetra*>& intact_tetras);



// Forward declarations for free MC-move functions defined in monte_carlo.cpp.
int try_flip_boundary_spin_MF_exact(MCSettings& mc, Spin* s);
int try_flip_worm(MCSettings& mc, Spin* root);
int try_flip_monopole_worm(MCSettings& mc, Tetra*tail_tetra, double target_length_mean=10);

struct MCStateMF {
    std::vector<Plaq*>    intact_plaqs;
    std::vector<Spin*>    classical_spins;  // type 4
    std::vector<Spin*>    boundary_spins;  // type 3
    std::vector<Tetra*>   class_tetras;   // fully-classical tetras (fixed after dilution)
    std::vector<QClusterMF> clusters;


    void partition_spins(std::vector<Spin>& spins);

    // UseExactBoundary=false: MF ⟨Sz⟩ estimate, no speculative re-diagonalisation.
    // UseExactBoundary=true:  exact eigenvalue shift via speculative re-diagonalisation.
    // if constexpr eliminates the branch at compile time.
    template<bool UseExactBoundary = false>
    void sweep(MCSettings& mc_);

    double energy();

    // utility (for benchmarking)
    // Classical-classical energy
    double classical_energy();
    // Onsite energy of all clusters
    double cluster_energy();
    // cluster-cluster meanfield energy
    double cluster_cluster_energy();
};

extern template void MCStateMF::sweep<false>(MCSettings&);
extern template void MCStateMF::sweep<true>(MCSettings&);

#pragma once

#include "quantum_cluster.hpp"
#include <random>
#include <set>
#include <sstream>

// total classical energy of one spin with its classical neighbours
inline double classical_bond_energy(const Spin* s, double J) {
    double E = 0;
    for (const Spin* nb : s->neighbours) {
        if (!nb->deleted && !nb->is_quantum())
            E += J * s->ising_val * nb->ising_val;
    }
    return E;
}

// ---- MC moves ----

struct MCSettings {
    double beta;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform{0.0, 1.0};

    size_t accepted_classical=0;
    size_t accepted_boundary=0;
    size_t accepted_quantum=0;

    size_t sweeps_attempted=0;

    void reset_acceptance(){
        accepted_classical=0;
        accepted_boundary=0;
        accepted_quantum=0;
        sweeps_attempted=0;
    }

    std::string acceptance(){
        std::ostringstream os;
        os << "Acc C="<< 100.0 * accepted_classical/sweeps_attempted<<"%\t"
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




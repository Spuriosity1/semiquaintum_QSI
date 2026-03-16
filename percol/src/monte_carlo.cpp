#include "monte_carlo.hpp"  

void MCState::partition_spins(std::vector<Spin>& spins){

    std::set<Spin*> boundary_spin_set;

    for (const auto& c : clusters){
        for (const auto s : c.spins){
            for (const auto nb : s->neighbours){
                if (!nb->deleted && nb->owning_cluster == nullptr) boundary_spin_set.insert(nb);
            }
        }
    }

    for (auto& s : spins){
        if (!s.deleted && !boundary_spin_set.contains(static_cast<Spin*>(&s))){
            classical_spins.push_back(&s);
        }
    }

    this->boundary_spins.reserve(boundary_spin_set.size());

    for (auto s : boundary_spin_set)   // local set
        this->boundary_spins.push_back(s);
}




//////////////////////////////////////////////////////////////////////
/// IMPORTANT PHYSICS: THE MC MOVES
///
/// There are four kinds of spins.
/// 1. deleted (trivial) spins, these do not count for anything.
/// 2. quantum spins. These are close enough to defects 
///    for the first order physics to be important.
/// 3. 'boundary' spins. These are classical neighbours of quantum spins. 
/// 4 . Fully classical spins with only classical neighbours.

// Move 1: flip a fully classical (type 4) spin
int try_flip_classical(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE = -2.0 * classical_bond_energy(s, Jzz); // flipping negates the bond energy
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);

    if (accept) s->ising_val *= -1;

    return accept;
}

// Move 2: change eigenstate of a cluster (no boundary change)
int try_flip_cluster_state(MCSettings& mc, QCluster& qc) {
    int old_idx = qc.eigenstate_idx;
    // propose a new eigenstate uniformly at random
    int new_idx = std::uniform_int_distribution<int>(0, qc.hilbert_dim() - 1)(mc.rng);
    double dE = qc.eigenvalues[new_idx] - qc.eigenvalues[old_idx];

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);
    if (accept) qc.eigenstate_idx = new_idx;
    return accept;
}

// Move 3: flip a boundary (type 3) spin
// This changes the cluster Hamiltonian, so we must re-diagonalise. 
// Unavoidable side-effect: change the physical meaning of the whole cluster.
//
// accept/reject on the combined ΔE_classical + ΔE_cluster
int try_flip_boundary_spin(MCSettings& mc, Spin* s) {
    // --- classical part: bonds to other classical spins ---
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);

    // --- quantum part: ΔE for each cluster this spin borders ---
    double dE_quantum = 0.0;
    // collect clusters affected and their new configs
    struct ClusterUpdate {
        QCluster*      qc;
        QCluster::BoundaryConfig new_config;
        Eigen::VectorXd new_eigenvalues;
        Eigen::MatrixXd new_eigenvectors;
        int             new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;


    for (Spin* nb : s->neighbours) {
        QCluster* qc = nb->owning_cluster;

        // skip if either a) spin is not quantum or b) cluster has already been processed
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate &u) { return u.qc == qc; }))
          continue;

        // find which boundary index s is in qc->boundary_spins
        int bidx = -1;
        for (int i = 0; i < (int)qc->boundary_spins.size(); i++)
            if (qc->boundary_spins[i] == s) { bidx = i; break; }
        if (bidx < 0) continue;

        QCluster::BoundaryConfig new_config = qc->boundary_config ^ (1u << bidx);
        double E_old = qc->energy();

        // tentatively diagonalise with new config
        QCluster tmp = *qc; // shallow copy is fine for this
        tmp.diagonalise(new_config);

        double E_new = tmp.eigenvalues[qc->eigenstate_idx];
        dE_quantum += E_new - E_old;

        updates.push_back({qc, new_config,
                           std::move(tmp.eigenvalues),
                           std::move(tmp.eigenvectors),
                           qc->eigenstate_idx});
    }

    double dE_total = dE_classical + dE_quantum;

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates) {
            u.qc->boundary_config  = u.new_config;
            u.qc->eigenvalues      = std::move(u.new_eigenvalues);
            u.qc->eigenvectors     = std::move(u.new_eigenvectors);
            u.qc->eigenstate_idx   = u.new_eigenstate_idx;
        }
    }
    return accept;
}




// Move 3a: flip a boundary (type 3) spin
// This changes the cluster Hamiltonian, so we must re-diagonalise. 
// Unavoidable side-effect: change the physical meaning of the whole cluster.
// We sample from Boltzmann of the updated cluster.
//
// accept/reject on the combined ΔE_classical + ΔE_cluster
int try_flip_boundary_spin_cluster_Boltzmann(MCSettings& mc, Spin* s) {
    // --- classical part: bonds to other classical spins ---
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);

    // --- quantum part: ΔF for each cluster this spin borders ---
    // uses free nergy in order to preseerve detailed blance
    double dF_quantum = 0.0;
    // collect clusters affected and their new configs
    struct ClusterUpdate {
        QCluster*      qc;
        QCluster::BoundaryConfig new_config;
        Eigen::VectorXd new_eigenvalues;
        Eigen::MatrixXd new_eigenvectors;
        int             new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;

    for (Spin* nb : s->neighbours) {
        QCluster* qc = nb->owning_cluster;
        // skip if either a) spin is not quantum or b) cluster has already been processed
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate &u) { return u.qc == qc; }))
          continue;

        // find which boundary index s is in qc->boundary_spins
        int bidx = -1;
        for (int i = 0; i < (int)qc->boundary_spins.size(); i++)
            if (qc->boundary_spins[i] == s) { bidx = i; break; }
        if (bidx < 0) continue;

        QCluster::BoundaryConfig new_config = qc->boundary_config ^ (1u << bidx);
        double betaF_old = -log(qc->partition_function(mc.beta));

        // tentatively diagonalise with new config
        QCluster tmp = *qc; // shallow copy is fine for this
        tmp.diagonalise(new_config);

        // choose new eigenstate from Boltzmann distribution on the updated cluster
        double Z = tmp.partition_function(mc.beta);
        double r = mc.uniform(mc.rng) * Z;
        double acc = 0;
        int new_idx = 0;
        for (; new_idx < tmp.eigenvalues.size(); new_idx++) {
            acc += std::exp(-mc.beta * tmp.eigenvalues[new_idx]);
            if (acc >= r) break;
        }

        double betaF_new = -log(tmp.partition_function(mc.beta));

        dF_quantum += (betaF_new - betaF_old)/mc.beta;

        updates.push_back({qc, new_config,
                           std::move(tmp.eigenvalues),
                           std::move(tmp.eigenvectors),
                           new_idx});
    }

    /* A note on detailed balance
     *
     * Express the energy of a configuration {z[], w[], n[]}===x of the 
     * classical spins, boundary spins and quantum spins respectively as 
     *                          E(z[], w[], n[]).
     * The target distribution is 
     *          p(z[], w[], n[]) = 1/Z exp(-beta * E(z[], w[], n[])
     * 
     * Proposal distribution P( x' | x )
     *
     * Move:
     * w[j]->-w[j] = w', n[Neigh(j)] -> n'[Neigh(j)]
     * P(w', n' | w, n) = 1/Zloc exp(-beta * sum(E_n'(w') - E_n(w)))
     *
     * where, for cluster j, Zloc(w)[j] = \sum_n exp(-beta E_n(w)[j] )
     *
     */

    double dE_total = dE_classical + dF_quantum;
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates) {
            u.qc->boundary_config  = u.new_config;
            u.qc->eigenvalues      = std::move(u.new_eigenvalues);
            u.qc->eigenvectors     = std::move(u.new_eigenvectors);
            u.qc->eigenstate_idx   = u.new_eigenstate_idx;
        }
    }
    return accept;
}


void MCState::sweep(MCSettings& mc_){
    mc_.sweeps_attempted++;
    for (auto s : classical_spins){
        mc_.accepted_classical += try_flip_classical(mc_, s);
    }
    for (auto s : boundary_spins){
        mc_.accepted_boundary += try_flip_boundary_spin(mc_, s);
    }
    for (auto& qc : clusters){
        mc_.accepted_quantum += try_flip_cluster_state(mc_, qc);
    }
}


////////////////////////////////////////////
/// IMPORTANT: THE GLOBAL ENERGY CALCULATION


double MCState::energy(){
    double E = 0;
    // very easy to get this wrong.
    
    double Jzz = ModelParams::get().Jzz;

    // Pure-classical energy: Ising ZZ interactions between type 3 and type 4 spins without double counting
    for (auto s : classical_spins){
        double acc = 0;
        for (auto nb : s->neighbours){
            if (!nb->deleted && !nb->is_quantum() && nb < s ) acc += nb->ising_val;
        }

        E += Jzz * acc * s->ising_val;
    }

    for (auto s : boundary_spins){
        double acc = 0;
        for (auto nb : s->neighbours){
            if (!nb->deleted && !nb->is_quantum() && nb < s ) acc += nb->ising_val;
        }

        E += Jzz * acc * s->ising_val;
    }

    // quantum energy (includes boundary - cluster interaction energy)
    for (const auto& qc : clusters){
        E += qc.energy();
    }

    return E;
}

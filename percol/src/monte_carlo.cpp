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

// Move 1a: flip a closed classical ring
int try_flip_ring(Plaq* p) {
    if (!p->is_complete) return false;
    // always accept if flippable, otherwise reject
    int prev_ising = p->member_spins[0]->ising_val;
    for (int i=1; i<6; ++i){
        auto s = p->member_spins[i];
        if(s->ising_val*prev_ising != -1) return 0;
        prev_ising = s->ising_val;
    }

    for (auto s : p->member_spins){
        s->ising_val *= -1;
    }
    return 1;
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
        int             new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;


    for (Spin* nb : s->neighbours) {
        QCluster* qc = static_cast<QCluster*>(nb->owning_cluster);

        // skip if either a) spin is not quantum or b) cluster has already been processed
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate &u) { return u.qc == qc; }))
          continue;

        // find which boundary index s is in qc->boundary_spins
        int bidx = -1;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i] == s) { bidx = i; break; }
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
                           qc->eigenstate_idx});
    }

    double dE_total = dE_classical + dE_quantum;

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates) {
            u.qc->boundary_config  = u.new_config;
            u.qc->eigenvalues      = std::move(u.new_eigenvalues);
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
        int             new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;

    for (Spin* nb : s->neighbours) {
        QCluster* qc = static_cast<QCluster*>(nb->owning_cluster);
        // skip if either a) spin is not quantum or b) cluster has already been processed
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate &u) { return u.qc == qc; }))
          continue;

        // find which boundary index s is in qc->boundary_spins
        int bidx = -1;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i] == s) { bidx = i; break; }
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
            u.qc->eigenstate_idx   = u.new_eigenstate_idx;
        }
    }
    return accept;
}


// Helper: MF interaction energy for cluster qc in eigenstate n
// Sums Jzz * <Sz_i(n)> * <Sz_j> over precomputed inter-cluster bonded pairs
static double mf_interaction(const QClusterMF& qc, int n) {
    double Jzz = ModelParams::get().Jzz;
    double E = 0;
    for (const auto& b : qc.mf_bonds)
        E += Jzz * qc.expect_Sz(n, b.my_site)
                 * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site);
    return E;
}

// MF Move 2: change eigenstate of a QClusterMF
int try_flip_cluster_state_MF(MCSettings& mc, QClusterMF& qc) {
    int old_idx = qc.eigenstate_idx;
    int new_idx = std::uniform_int_distribution<int>(0, qc.hilbert_dim() - 1)(mc.rng);
    double dE_internal = qc.eigenvalues[new_idx] - qc.eigenvalues[old_idx];
    double dE_mf       = mf_interaction(qc, new_idx) - mf_interaction(qc, old_idx);

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * (dE_internal + dE_mf));
    if (accept) qc.eigenstate_idx = new_idx;
    return accept;
}


// MF Move 3: flip a classical boundary spin adjacent to QClusterMF clusters
int try_flip_boundary_spin_MF(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);

    double dE_quantum = 0.0;
    struct ClusterUpdate {
        QClusterMF* qc;
        QClusterMF::BoundaryConfig new_config;
        QClusterMF tmp;  // tentative diagonalisation result — moved in on accept
    };
    std::vector<ClusterUpdate> updates;
    updates.reserve(6);

    for (Spin* nb : s->neighbours) {
        QClusterMF* qc = static_cast<QClusterMF*>(nb->owning_cluster);
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate& u) { return u.qc == qc; }))
            continue;

        int bidx = -1;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i] == s) { bidx = i; break; }
        if (bidx < 0) continue;

        QClusterMF::BoundaryConfig new_config = qc->boundary_config ^ (1u << bidx);
        double E_old = qc->energy();
        double mf_old = mf_interaction(*qc, qc->eigenstate_idx);

        QClusterMF tmp = *qc;
        tmp.diagonalise(new_config);

        double E_new = tmp.eigenvalues[qc->eigenstate_idx];
        double mf_new = mf_interaction(tmp, qc->eigenstate_idx);
        dE_quantum += (E_new - E_old) + (mf_new - mf_old);

        updates.push_back({qc, new_config, std::move(tmp)});
    }

    double dE_total = dE_classical + dE_quantum;
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates)
            *u.qc = std::move(u.tmp);  // installs precomputed eigenvalues + Sz_expect
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


void MCStateMF::partition_spins(std::vector<Spin>& spins) {
    std::set<Spin*> boundary_spin_set;

    for (const auto& c : clusters) {
        for (const auto s : c.spins) {
            for (const auto nb : s->neighbours) {
                if (!nb->deleted && nb->owning_cluster == nullptr)
                    boundary_spin_set.insert(nb);
            }
        }
    }

    for (auto& s : spins) {
        if (!s.deleted && !boundary_spin_set.contains(static_cast<Spin*>(&s))) {
            classical_spins.push_back(&s);
        }
    }

    this->boundary_spins.reserve(boundary_spin_set.size());
    for (auto s : boundary_spin_set)
        this->boundary_spins.push_back(s);
}


void MCStateMF::sweep(MCSettings& mc_) {
    mc_.sweeps_attempted++;
    for (auto p : intact_plaqs) {
        mc_.accepted_plaq += try_flip_ring(p);
    }
    for (auto s : classical_spins) {
        mc_.accepted_classical += try_flip_classical(mc_, s);
    }
    for (auto s : boundary_spins) {
        mc_.accepted_boundary += try_flip_boundary_spin_MF(mc_, s);
    }
    for (auto& qc : clusters) {
        mc_.accepted_quantum += try_flip_cluster_state_MF(mc_, qc);
    }
}


double MCStateMF::energy() {
    double E = 0;
    double Jzz = ModelParams::get().Jzz;

    // Pure-classical energy: Ising ZZ interactions between type 3 and type 4 spins without double counting
    for (auto s : classical_spins) {
        double acc = 0;
        for (auto nb : s->neighbours) {
            if (!nb->deleted && !nb->is_quantum() && nb < s) acc += nb->ising_val;
        }
        E += Jzz * acc * s->ising_val;
    }

    for (auto s : boundary_spins) {
        double acc = 0;
        for (auto nb : s->neighbours) {
            if (!nb->deleted && !nb->is_quantum() && nb < s) acc += nb->ising_val;
        }
        E += Jzz * acc * s->ising_val;
    }

    // cluster eigenvalues (includes classical and quantum boundary couplings)
    for (const auto& qc : clusters) {
        E += qc.energy();
    }

    // MF cross-terms: factor 1/2 to avoid double-counting (each bond in mf_bonds from both sides)
    for (const auto& qc : clusters) {
        for (const auto& b : qc.mf_bonds) {
            E += 0.5 * Jzz * qc.expect_Sz(qc.eigenstate_idx, b.my_site)
                           * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site);
        }
    }

    return E;
}

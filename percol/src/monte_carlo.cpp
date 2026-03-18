#include "monte_carlo.hpp"
#include <unordered_map>

void MCState::partition_spins(std::vector<Spin>& spins){

    std::set<Spin*> boundary_spin_set;

    auto collect_boundaries = [&](auto& clusters) {
        for (const auto& c : clusters) {
            for (const auto s : c.spins) {
                for (const auto nb : s->neighbours) {
                    if (!nb->deleted && nb->owning_cluster == nullptr)
                        boundary_spin_set.insert(nb);
                }
            }
        }
    };
    collect_boundaries(exact_clusters);
    collect_boundaries(mf_clusters);

    for (auto& s : spins){
        if (!s.deleted && !s.is_quantum() && !boundary_spin_set.contains(static_cast<Spin*>(&s))){
            classical_spins.push_back(&s);
        }
    }

    this->boundary_spins.reserve(boundary_spin_set.size());
    for (auto s : boundary_spin_set)
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
/// 4. Fully classical spins with only classical neighbours.

// Move 1: flip a fully classical (type 4) spin
int try_flip_classical(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE = -2.0 * classical_bond_energy(s, Jzz);
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);

    if (accept) s->ising_val *= -1;

    return accept;
}

// Move 2a: change eigenstate of an exact cluster (no boundary change, no propagation)
int try_flip_cluster_state(MCSettings& mc, QCluster& qc) {
    int old_idx = qc.eigenstate_idx();
    int new_idx = std::uniform_int_distribution<int>(0, qc.hilbert_dim() - 1)(mc.rng);
    double dE = qc.eigenvalue(new_idx) - qc.eigenvalue(old_idx);

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);
    if (accept) qc.set_eigenstate(new_idx);
    return accept;
}

// Move 2b: change eigenstate of an MF cluster, propagating updated mean fields
// to any neighbouring QClusterMF whose boundary configs depend on this cluster.
int try_flip_cluster_state(MCSettings& mc, QClusterMF& qc) {
    int old_idx = qc.eigenstate_idx();
    int new_idx = std::uniform_int_distribution<int>(0, qc.hilbert_dim() - 1)(mc.rng);
    double dE = qc.eigenvalue(new_idx) - qc.eigenvalue(old_idx);

    // Build updated boundary configs for each neighbouring cluster whose field changes.
    std::unordered_map<QClusterBase*, QClusterBase::BoundaryConfig> new_configs;
    for (int si = 0; si < qc.n_spins(); si++) {
        double old_sz = qc.sz_expectation_at(si, old_idx);
        double new_sz = qc.sz_expectation_at(si, new_idx);
        if (old_sz == new_sz) continue;
        for (Spin* nb : qc.spins[si]->neighbours) {
            QClusterBase* B = nb->owning_cluster;
            if (!B || B == &qc) continue;
            int bidx = B->boundary_index(qc.spins[si]);
            if (bidx < 0) continue;
            if (!new_configs.count(B)) new_configs[B] = B->boundary_config();
            new_configs[B][bidx] = new_sz;
        }
    }

    // Stage neighbour diagonalisations and accumulate ΔE.
    struct NeighborUpdate { QClusterBase* B; QClusterBase::Snapshot snap; int kept_state; };
    std::vector<NeighborUpdate> updates;
    for (auto& [B, cfg] : new_configs) {
        auto snap = B->propose_with_config(cfg);
        dE += snap.energy_at(B->eigenstate_idx()) - B->energy();
        updates.push_back({B, std::move(snap), B->eigenstate_idx()});
    }

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);
    if (accept) {
        qc.set_eigenstate(new_idx);
        for (auto& u : updates)
            u.B->commit(std::move(u.snap), u.kept_state);
    }
    return accept;
}

// Move 3: flip a boundary (type 3) spin.
// This changes the cluster Hamiltonian, so we must re-diagonalise.
// accept/reject on the combined ΔE_classical + ΔE_cluster
int try_flip_boundary_spin(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);
    double dE_quantum = 0.0;

    struct ClusterUpdate {
        QClusterBase*          qc;
        QClusterBase::Snapshot snap;
        int                    new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;

    for (Spin* nb : s->neighbours) {
        QClusterBase* qc = nb->owning_cluster;

        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate& u) { return u.qc == qc; }))
            continue;

        int bidx = qc->boundary_index(s);
        if (bidx < 0) continue;

        double E_old = qc->energy();
        auto snap = qc->propose_boundary_flip(bidx);
        double E_new = snap.energy_at(qc->eigenstate_idx());
        dE_quantum += E_new - E_old;

        updates.push_back({qc, std::move(snap), qc->eigenstate_idx()});
    }

    double dE_total = dE_classical + dE_quantum;

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates)
            u.qc->commit(std::move(u.snap), u.new_eigenstate_idx);
    }
    return accept;
}




// Move 3a: boundary flip with Boltzmann sampling of the updated cluster state.
int try_flip_boundary_spin_cluster_Boltzmann(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);
    double dF_quantum = 0.0;

    struct ClusterUpdate {
        QClusterBase*          qc;
        QClusterBase::Snapshot snap;
        int                    new_eigenstate_idx;
    };
    std::vector<ClusterUpdate> updates;

    for (Spin* nb : s->neighbours) {
        QClusterBase* qc = nb->owning_cluster;
        if (!qc ||
            std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate& u) { return u.qc == qc; }))
            continue;

        int bidx = qc->boundary_index(s);
        if (bidx < 0) continue;

        double betaF_old = -std::log(qc->partition_function(mc.beta));
        auto snap = qc->propose_boundary_flip(bidx);

        int new_idx = snap.sample_boltzmann(mc.beta, mc.uniform(mc.rng));

        double betaF_new = -std::log(snap.partition_function(mc.beta));
        dF_quantum += (betaF_new - betaF_old) / mc.beta;

        updates.push_back({qc, std::move(snap), new_idx});
    }

    double dE_total = dE_classical + dF_quantum;
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE_total);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates)
            u.qc->commit(std::move(u.snap), u.new_eigenstate_idx);
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
    for (auto& qc : exact_clusters){
        mc_.accepted_quantum += try_flip_cluster_state(mc_, qc);
    }
    for (auto& qc : mf_clusters){
        mc_.accepted_quantum += try_flip_cluster_state(mc_, qc);
    }
}


////////////////////////////////////////////
/// IMPORTANT: THE GLOBAL ENERGY CALCULATION


double MCState::energy(){
    double E = 0;
    double Jzz = ModelParams::get().Jzz;

    // Pure-classical energy: bonds between type 3 and type 4 spins, no double-counting
    for (auto s : classical_spins){
        double acc = 0;
        for (auto nb : s->neighbours){
            if (!nb->deleted && !nb->is_quantum() && nb < s) acc += nb->ising_val;
        }
        E += Jzz * acc * s->ising_val;
    }

    for (auto s : boundary_spins){
        double acc = 0;
        for (auto nb : s->neighbours){
            if (!nb->deleted && !nb->is_quantum() && nb < s) acc += nb->ising_val;
        }
        E += Jzz * acc * s->ising_val;
    }

    // quantum energy (includes boundary–cluster interaction)
    for (const auto& qc : exact_clusters) E += qc.energy();
    for (const auto& qc : mf_clusters)    E += qc.energy();

    return E;
}

#include "random"
#include "monte_carlo.hpp"
#include <unordered_set>

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
        if (!s.deleted && !s.is_quantum() && !boundary_spin_set.contains(static_cast<Spin*>(&s))){
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

// Move 1: single-spin Metropolis on a fully classical (type 4) spin.
// ΔE = −2 * Jzz * s * Σ_{classical neighbours} nb   (flipping negates all bonds).
int try_flip_classical(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE = -2.0 * classical_bond_energy(s, Jzz); // flipping negates the bond energy
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);

    if (accept) s->ising_val *= -1;

    return accept;
}

// Move 1a: simultaneous flip of all 6 spins on a complete hexagonal plaquette.
// These are zero-energy moves within the classical ice manifold (alternating
// ↑↓↑↓↑↓ pattern has zero net bond energy change on flip).  Essential for
// ergodicity: without ring moves the classical sector can get trapped.
// Always accepted when the plaquette is in the alternating state; rejected otherwise.
int try_flip_ring(Plaq* p) {
    if (!p->is_complete) return 0;
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


// Move 1b: simultaneous flip of a worm.
//
// Finds a closed loop of classical (and/or boundary) spins and flips them all.
// Zero-energy because at every tetrahedron in the path the two loop spins have
// *opposite* ising_val (heal_val alternates ±), so (s_in + s_out) = 0 and each
// tetra's contribution to the crossing-bond energy cancels.
//
// Boundary spins are allowed on the path.  When two boundary spins of the same
// cluster appear on the worm they must carry opposite heal_val values, so their
// net change to H_boundary also cancels — the cluster eigenvalues are unchanged.
// We still call diagonalise() after the worm to keep boundary_config consistent
// for future boundary-spin Metropolis moves; this is a cache lookup (O(1)).
int try_flip_worm(MCSettings& mc, Spin* root) {
    if (root->deleted || root->is_quantum()) return 0;
    if (!root->owning_tetras[0] || !root->owning_tetras[1]) return 0;

    Tetra* const tail_tetra = root->owning_tetras[0];
    Tetra* head_tetra       = root->owning_tetras[1];

    const int root_val = root->ising_val;
    std::vector<Spin*> path = {root};
    root->ising_val *= -1;
    Spin* prev_spin = root;

    // heal_val: the ising_val a candidate must currently have so that flipping it
    // heals the defect at head_tetra and keeps (s_in + s_out) = 0 per tetra.
    // Alternates every step: -root_val, +root_val, -root_val, …
    int heal_val = -root_val;

    constexpr int MAX_STEPS = 200;

    while (head_tetra != tail_tetra) {
        if ((int)path.size() > MAX_STEPS) {
            for (Spin* s : path) s->ising_val *= -1;
            return 0;
        }

        std::vector<Spin*> candidates;
        for (Spin* s : head_tetra->member_spins) {
            if (s == prev_spin || s->deleted || s->is_quantum()) continue;
            if (s->ising_val != heal_val) continue;
            candidates.push_back(s);
        }

        if (candidates.empty()) {
            for (Spin* s : path) s->ising_val *= -1;
            return 0;
        }

        Spin* next = candidates[
            std::uniform_int_distribution<int>(0, (int)candidates.size()-1)(mc.rng)];

        Tetra* next_head = (next->owning_tetras[0] == head_tetra)
                            ? next->owning_tetras[1]
                            : next->owning_tetras[0];
        if (!next_head) {
            for (Spin* s : path) s->ising_val *= -1;
            return 0;
        }

        next->ising_val *= -1;
        path.push_back(next);
        prev_spin  = next;
        heal_val   = -heal_val;
        head_tetra = next_head;
    }

    // Validate: heal_val must equal root_val (odd number of steps taken) so that
    // the defect propagated back to tail_tetra exactly cancels the original defect.
    if (heal_val != root_val) {
        for (Spin* s : path) s->ising_val *= -1;
        return 0;
    }

    // Keep boundary_config consistent for any quantum clusters whose boundary spins
    // were flipped.  diagonalise() is a cache lookup here — O(1) for small clusters.
    std::unordered_set<QClusterBase*> touched;
    for (Spin* s : path) {
        for (Spin* nb : s->neighbours) {
            if (!nb->deleted && nb->is_quantum() && nb->owning_cluster)
                touched.insert(nb->owning_cluster);
        }
    }
    for (QClusterBase* qcb : touched) {
        auto* qc = static_cast<QClusterMF*>(qcb);
        QClusterMF::BoundaryConfig new_cfg = 0;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i]->ising_val == +1)
                new_cfg |= (1u << i);
        qc->diagonalise(new_cfg);
    }

    return 1;  // loop closed, all flips committed
}

// NB does NOT include usual sublattice factor
int tetra_charge(const Tetra* t){
    int q = 0;
    for (Spin* s : t->member_spins) q += s->ising_val;
    return q;
}

// filters out a list of tetras with monopoles on them
std::vector<Tetra*> find_monopole_tetras(const std::vector<Tetra*>& intact_tetras){
    std::vector<Tetra*> retval;
    for (auto t : intact_tetras) {
        if (tetra_charge(t) != 0) { retval.push_back(t); }
    }
    return retval;
}


// Move 1c: monopole worm — move a monopole through the ice manifold.
//
// Searches for an "intact" tetrahedron with a non-zero charge
// (Q = Σ ising_val ≠ 0, i.e. a monopole; intact = all 4 member spins non-deleted,
// non-quantum).  Picks one at random as the worm tail, then walks an open, non-closed
// path nose-to-tail using the same alternating heal_val rule as the zero-energy worm (1b):
// at step k the chosen spin must have ising_val == heal_val to restore ice-rule at the
// current head and propagate the charge to the next tetrahedron.
//
// The walk terminates when either
//   (a) no nose-to-tail candidate exists at the current head (stuck in the lattice), or
//   (b) the charge arriving at the new head cancels a pre-existing monopole there
//       (Q_head = 0 after the flip — "annihilation"), giving ΔE ≈ 0 in the ice manifold.
//
// The whole path is accepted or rejected with a single Metropolis step on the total
// accumulated ΔE (classical Jzz bonds + quantum eigenvalue shift via tentative
// re-diagonalisation of any touched QClusterMF).
int try_flip_monopole_worm(MCSettings& mc, Tetra*tail_tetra, double target_length_mean) {
    // For all intact tetrahedra that currently carry a monopole charge, try to move them
    // intact_tetras is fixed at dilution time (all 4 members present and classical);
    // only Q = Σ ising_val changes dynamically, so we scan the pre-built list.
    int q_tail = tetra_charge(tail_tetra);
    if (q_tail == 0) return 0; // refuse

    // heal_val for the first step matches sign(Q_tail): flipping a spin with
    // ising_val == heal_val moves the charge out of tail_tetra (Q_tail → 0 if |Q|=2)
    // and creates a charge of opposite sign in the next tetrahedron.
    int heal_val = (q_tail > 0) ? +1 : -1; // majority spin type, flippable type

    std::vector<Spin*> path;
    Tetra* head_tetra = tail_tetra;
    Spin* prev_spin  = nullptr;
    const int target_length = std::poisson_distribution<int>(target_length_mean)(mc.rng);

    while ((int)path.size() < target_length) {
        // nose-to-tail candidates: non-deleted, non-quantum spins in head with
        // ising_val == heal_val (not the spin we just came through)
        std::vector<Spin*> candidates;
        for (Spin* s : head_tetra->member_spins) {
            if (s == prev_spin || s->deleted || s->is_quantum()) continue;
            if (s->ising_val == heal_val) candidates.push_back(s);
        }
        if (candidates.empty()) break;   // stuck — accept/reject current path

        Spin* next_spin = candidates[
            std::uniform_int_distribution<int>(0, (int)candidates.size() - 1)(mc.rng)
        ];
        Tetra* next_head = (next_spin->owning_tetras[0] == head_tetra)
                            ? next_spin->owning_tetras[1]
                            : next_spin->owning_tetras[0];
        assert(next_head);

        next_spin->ising_val *= -1;
        path.push_back(next_spin);
        prev_spin = next_spin;
        head_tetra      = next_head;
        heal_val  = -heal_val;

    }

    if (path.empty()) return 0;

    // If the final head contains a quantum spin, the last flip is an uncompensated
    // boundary-config change whose energy we don't include in dE.  Backtrack one step.
    for (Spin* m : head_tetra->member_spins) {
        if (m->is_quantum()) {
            path.back()->ising_val *= -1;
            Tetra* prev = (path.back()->owning_tetras[0] == head_tetra)
                            ? path.back()->owning_tetras[1]
                            : path.back()->owning_tetras[0];
            path.pop_back();
            head_tetra = prev;
            if (path.empty()) return 0;
            break;
        }
    }

    // ΔE is purely from head and tail tetras (all intermediate tetras return to ice rule).
    // Using sigma convention: E(tetra) = Jzz/2 * (Q²-4), so ΔE = Jzz/2 * ΔQ²
    const double Jzz = ModelParams::get().Jzz;
    int Q_tail_proposed = tetra_charge(tail_tetra);
    // q_head is current charge at head; undo the last flip to get initial charge
    int Q_head_proposed = tetra_charge(head_tetra);
    int Q_head_initial = Q_head_proposed - 2 * path.back()->ising_val;
    double dE = (Jzz / 2.0) * (
        (double)(Q_tail_proposed * Q_tail_proposed - q_tail * q_tail) +
        (double)(Q_head_proposed * Q_head_proposed - Q_head_initial * Q_head_initial)
    );

    // --- Metropolis ---
    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * (dE));
    if (!accept) {
        for (Spin* s : path) s->ising_val *= -1;
        return 0;
    }


    // Update boundary configs of any quantum clusters adjacent to flipped spins.
    // The worm is a purely classical move: the quantum cluster should follow
    // adiabatically — staying on the eigenstate whose energy is closest to its
    // pre-worm energy in the new Hamiltonian (adiabatic following).
    std::unordered_set<QClusterMF*> touched;
    for (Spin* s : path)
        for (Spin* nb : s->neighbours)
            if (!nb->deleted && nb->is_quantum() && nb->owning_cluster)
                touched.insert(static_cast<QClusterMF*>(nb->owning_cluster));
    for (QClusterMF* qc : touched) {
        const double E_old = qc->energy();   // eigenvalue before boundary change

        QClusterMF::BoundaryConfig new_cfg = 0;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i]->ising_val == +1)
                new_cfg |= (1u << i);
        qc->diagonalise(new_cfg);

        // Find eigenstate in new basis with energy closest to E_old.
        int best_idx = 0;
        double best_dist = std::abs(qc->eigenvalues[0] - E_old);
        for (int n = 1; n < (int)qc->eigenvalues.size(); n++) {
            double d = std::abs(qc->eigenvalues[n] - E_old);
            if (d < best_dist) { best_dist = d; best_idx = n; }
        }
        qc->eigenstate_idx = best_idx;
    }

    return 1;
}


// Move 2 (QCluster): propose a uniformly random new eigenstate within a cluster.
// The boundary config (and therefore the spectrum) is unchanged; only eigenstate_idx moves.
// ΔE = eigenvalues[new] − eigenvalues[old].  Metropolis accept.
int try_flip_cluster_state(MCSettings& mc, QCluster& qc) {
    int old_idx = qc.eigenstate_idx;
    // propose a new eigenstate uniformly at random
    int new_idx = std::uniform_int_distribution<int>(0, qc.hilbert_dim() - 1)(mc.rng);
    double dE = qc.eigenvalues[new_idx] - qc.eigenvalues[old_idx];

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);
    if (accept) qc.eigenstate_idx = new_idx;
    return accept;
}

// Move 3 (QCluster): Metropolis flip of a boundary (type 3) spin.
// Flipping s changes the Ising field seen by every adjacent cluster, which
// shifts the entire cluster spectrum.  Steps:
//   1. Classical ΔE from bonds to other classical/boundary spins.
//   2. For each adjacent cluster: tentatively re-diagonalise with the new
//      boundary config; ΔE_cluster = eigenvalues_new[old_idx] − eigenvalues_old[old_idx].
//      (The eigenstate index is preserved across the boundary flip.)
//   3. Global Metropolis on ΔE_classical + Σ ΔE_cluster.
//   4. On accept: install new boundary_config and eigenvalues into each cluster.
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


// Mean-field inter-cluster coupling energy for cluster qc in eigenstate n.
// Approximates the Jzz bond between quantum spins in different clusters as
//   Jzz * <Sz_i>_n * <Sz_j>_{eigenstate of other cluster}
// summed over all cross-cluster nearest-neighbour pairs listed in mf_bonds.
// Note: each bond is listed from both sides, so MCStateMF::energy() applies
// a factor of 0.5 to avoid double-counting.
static double mf_interaction(const QClusterMF& qc, int n) {
    double Jzz = ModelParams::get().Jzz;
    double E = 0;
    for (const auto& b : qc.mf_bonds)
        E += Jzz * qc.expect_Sz(n, b.my_site)
                 * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site);
    return E;
}

// Move 2 (QClusterMF): Gibbs (Boltzmann) sample a new eigenstate.
// Samples directly from exp(-β*(E_n + E_MF_n)), where E_MF_n = mf_interaction(qc, n)
// uses the current eigenstate of neighbouring clusters (mean-field approximation).
// Changing eigenstate_idx alters <Sz> exported to neighbouring clusters,
// but those clusters' mf_interaction is only re-evaluated on their next move
// (mean-field is not self-consistent within a sweep).
// Gibbs sampling avoids the self-loop bias of uniform Metropolis: acceptance → 0
// as T → 0 when the ground state is unique.
int try_flip_cluster_state_MF(MCSettings& mc, QClusterMF& qc) {
    int old_idx = qc.eigenstate_idx;
    int dim = qc.hilbert_dim();

    double Z = 0;
    for (int n = 0; n < dim; n++)
        Z += std::exp(-mc.beta * (qc.eigenvalues[n] + mf_interaction(qc, n)));

    double r = mc.uniform(mc.rng) * Z;
    double acc = 0;
    int new_idx = dim - 1;
    for (int n = 0; n < dim; n++) {
        acc += std::exp(-mc.beta * (qc.eigenvalues[n] + mf_interaction(qc, n)));
        if (acc >= r) { new_idx = n; break; }
    }

    qc.eigenstate_idx = new_idx;
    return new_idx != old_idx ? 1 : 0;
}


// Move 3 (QClusterMF): Metropolis flip of a boundary spin — exact version.
// Re-diagonalises each affected cluster speculatively to get the exact eigenvalue shift.
// On accept: *qc = move(tmp) installs precomputed eigenvalues + Sz_expect atomically.
// Active when MCStateMF::sweep<true> is used; the cheaper MF version is sweep<false>.
int try_flip_boundary_spin_MF_exact(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);

    double dE_quantum = 0.0;
    struct ClusterUpdate {
        QClusterMF* qc;
        QClusterMF::BoundaryConfig new_config;
        QClusterMF tmp;
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
            *u.qc = std::move(u.tmp);
    }
    return accept;
}

// Move 3 (QClusterMF): Metropolis flip of a boundary spin adjacent to QClusterMF clusters.
// ΔE_classical: bonds to other classical/boundary spins (exact).
// ΔE_cluster:   Jzz * (−2σ_s) * Σ_{cluster spins nb} ⟨Sz_nb⟩  (MF approximation).
//   This avoids speculative re-diagonalisation; the boundary coupling is handled the
//   same way as cross-cluster MF bonds.  On accept, diagonalise() updates the cluster
//   eigenvalues and Sz_expect via an O(1) cache lookup.
//   For the exact version (speculative re-diag) see try_flip_boundary_spin_MF_exact above.
int try_flip_boundary_spin_MF(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE = -2.0 * classical_bond_energy(s, Jzz);

    struct ClusterUpdate {
        QClusterMF* qc;
        QClusterMF::BoundaryConfig new_config;
    };
    std::vector<ClusterUpdate> updates;
    updates.reserve(6);

    for (Spin* nb : s->neighbours) {
        QClusterMF* qc = static_cast<QClusterMF*>(nb->owning_cluster);
        if (!qc) continue;

        int site_nb = qc->spin_index(nb);
        if (site_nb < 0) continue;

        // MF estimate: flipping s changes its coupling to each adjacent cluster spin
        dE += Jzz * (-2.0 * s->ising_val) * qc->expect_Sz(qc->eigenstate_idx, site_nb);

        if (std::any_of(updates.begin(), updates.end(),
                        [&](const ClusterUpdate& u) { return u.qc == qc; }))
            continue;

        int bidx = -1;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i] == s) { bidx = i; break; }
        if (bidx < 0) continue;

        updates.push_back({qc, qc->boundary_config ^ (1u << bidx)});
    }

    int accept = mc.uniform(mc.rng) < std::exp(-mc.beta * dE);
    if (accept) {
        s->ising_val *= -1;
        for (auto& u : updates)
            u.qc->diagonalise(u.new_config);  // O(1) cache lookup; updates eigenvalues + Sz_expect + boundary_config
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
        if (!s.deleted && !s.is_quantum() && !boundary_spin_set.contains(static_cast<Spin*>(&s))) {
            classical_spins.push_back(&s);
        }
    }

    this->boundary_spins.reserve(boundary_spin_set.size());
    for (auto s : boundary_spin_set)
        this->boundary_spins.push_back(s);

    find_intact_tetras(classical_spins, intact_tetras);
}



void find_intact_tetras(const std::vector<Spin*>& classical_spins,
        std::vector<Tetra*>& intact_tetras)
{

    // Build the static list of fully-classical tetrahedra (intact = no deleted or quantum
    // member spins).  Q values change during the simulation but intact status does not.
    intact_tetras.resize(0);
    std::unordered_set<Tetra*> seen_tetras;
    for (Spin* s : classical_spins) {
        for (Tetra* t : s->owning_tetras) {
            if (!t || !seen_tetras.insert(t).second) continue;
            bool intact = true;
            for (Spin* m : t->member_spins)
                if (m->deleted || m->is_quantum()) { intact = false; break; }
            if (intact) intact_tetras.push_back(t);
        }
    }
}


template<bool UseExactBoundary>
void MCStateMF::sweep(MCSettings& mc_) {
    mc_.sweeps_attempted++;
    if (mc_.moves & MOVE_RING) {
        for (auto p : intact_plaqs)
            mc_.accepted_plaq += try_flip_ring(p);
    }
    if (mc_.moves & MOVE_CLASSICAL) {
        for (auto s : classical_spins)
            mc_.accepted_classical += try_flip_classical(mc_, s);
    }
    if (mc_.moves & MOVE_WORM) {
        auto s = classical_spins[
            std::uniform_int_distribution<int>(0, (int)classical_spins.size()-1)(mc_.rng)
        ];
        mc_.accepted_worm += try_flip_worm(mc_, s);
    }
    if (mc_.moves & MOVE_MONOPOLE) {
        auto monopoles = find_monopole_tetras(intact_tetras);
        if (!monopoles.empty()) {
            mc_.attempted_monopole += monopoles.size();
            // guess the length to go for: (N_classical / N_mon)^(1/3) ~ mean free path
            double monopole_mfp = std::pow(
                (double)classical_spins.size() / monopoles.size(), 1.0/3.0);
            for (auto& t : monopoles)
                mc_.accepted_monopole += try_flip_monopole_worm(mc_, t, monopole_mfp);
        }
    }
    if (mc_.moves & MOVE_BOUNDARY) {
        for (auto s : boundary_spins) {
            if constexpr (UseExactBoundary)
                mc_.accepted_boundary += try_flip_boundary_spin_MF_exact(mc_, s);
            else
                mc_.accepted_boundary += try_flip_boundary_spin_MF(mc_, s);
        }
    }
    if (mc_.moves & MOVE_QUANTUM) {
        for (auto& qc : clusters)
            mc_.accepted_quantum += try_flip_cluster_state_MF(mc_, qc);
    }
}

template void MCStateMF::sweep<false>(MCSettings&);
template void MCStateMF::sweep<true>(MCSettings&);


// Total energy of the system.  Three additive contributions:
//
//  1. Classical ZZ bonds between type-3 (boundary) and type-4 (classical) spins.
//     Pointer ordering (nb < s) avoids double-counting.
//     Bonds to quantum spins are excluded — those are inside the cluster eigenvalue.
//
//  2. Cluster eigenvalues: each cluster contributes eigenvalues[eigenstate_idx],
//     which already includes the Jzz coupling to its classical boundary spins.
//
//  3. MF cross-terms: 0.5 * Jzz * Σ_{MF bonds} <Sz_i> * <Sz_j>.
//     The 0.5 cancels the double-counting arising from each bond appearing in
//     both clusters' mf_bonds list.
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

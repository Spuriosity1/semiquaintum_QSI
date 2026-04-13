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

static const char* spin_type(const Spin* s) {
    if (s->deleted)      return "deleted";
    if (s->is_quantum()) return "quantum";
    for (Spin* nb : s->neighbours)
        if (!nb->deleted && nb->is_quantum()) return "boundary";
    return "classical";
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

// Centralised accept-reject used by all moves.
// Canonical mode (mc.muca == nullptr): Boltzmann factor exp(−β dE) × hastings.
// MUCA mode     (mc.muca != nullptr): exp(lnG[k_old] − lnG[k_new]) × hastings.
// apply_fn() commits the state change; mc.muca->E_current is updated on accept.
// Returns 1 on accept, 0 on reject.  Out-of-range energy → always reject.
template<typename ApplyFn>
static int muca_accept(MCSettings& mc, double dE, double hastings, ApplyFn&& apply_fn) {
    double log_acc;
    double E_new = mc.muca ? mc.muca->E_current + dE : 0.0;
    if (mc.muca == nullptr) {
        log_acc = -mc.beta * dE + std::log(hastings);
    } else {
        int k_old = mc.muca->energy_bin(mc.muca->E_current);
        int k_new = mc.muca->energy_bin(E_new);
        if (k_old < 0 || k_new < 0) return 0;   // reflecting boundary
        log_acc = mc.muca->lnG[k_old] - mc.muca->lnG[k_new] + std::log(hastings);
    }
    if (mc.uniform(mc.rng) < std::exp(log_acc)) {
        apply_fn();
        if (mc.muca) mc.muca->E_current = E_new;
        return 1;
    }
    return 0;
}

// Forward declaration: defined later, used by try_flip_ring.
static double mf_interaction(const QClusterMF& qc, int n);

// Move 1: single-spin Metropolis on a fully classical (type 4) spin.
// ΔE = −2 * Jzz * s * Σ_{classical neighbours} nb   (flipping negates all bonds).
int try_flip_classical(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE = -2.0 * classical_bond_energy(s, Jzz);
    return muca_accept(mc, dE, 1.0, [&]{ s->ising_val *= -1; });
}

// Move 1a: simultaneous flip of all 6 spins on a complete hexagonal plaquette.
// These are zero-energy moves within the classical ice manifold (alternating
// ↑↓↑↓↑↓ pattern has zero net bond energy change on flip).  Essential for
// ergodicity: without ring moves the classical sector can get trapped.
// Always accepted when the plaquette is in the alternating state; rejected otherwise.
int try_flip_ring_vibe(Plaq* p, MCSettings& mc) {
    static const double ENERGY_TOL = 1e-9;

    if (!p->is_complete) return 0;

    // Check alternating pattern
    int prev_ising = p->member_spins[0]->ising_val;
    for (int i = 1; i < 6; ++i) {
        if (p->member_spins[i]->ising_val * prev_ising != -1) return 0;
        prev_ising = p->member_spins[i]->ising_val;
    }

    // Collect quantum clusters adjacent to flipped spins
    std::unordered_set<QClusterMF*> touched;
    for (int i = 0; i < 6; i++) {
        Spin* s = p->member_spins[i];
        for (Spin* nb : s->owning_tetras[i%2]->member_spins)
            if (!nb->deleted && nb->is_quantum() && nb->owning_cluster)
                touched.insert(static_cast<QClusterMF*>(nb->owning_cluster));
    }

    // Fast path: no quantum neighbours — ring is purely classical, always accept
    if (touched.empty()) {
        for (auto s : p->member_spins) s->ising_val *= -1;
        return 1;
    }

    // MF neighbors of touched clusters that are not themselves touched.
    // We need their mf_interaction in the local energy sum because their bonds
    // to touched clusters appear in their mf_bonds lists (from the other side).
    std::unordered_set<QClusterMF*> mf_neighbors;
    for (QClusterMF* qc : touched)
        for (const auto& b : qc->mf_bonds)
            if (!touched.count(b.other))
                mf_neighbors.insert(b.other);

    // Local MF energy: sum over touched ∪ mf_neighbors, with 0.5 to avoid
    // double-counting (every bond appears once in each endpoint's mf_bonds list).
    // This equals ΔE_cc exactly because:
    //   - bonds inside touched: listed in both touched sums → factor 2 × 0.5 = 1 ✓
    //   - bonds touched↔neighbor: listed once each side → factor 2 × 0.5 = 1 ✓
    //   - bonds inside mf_neighbors: both sides unchanged → cancel in ΔE ✓
    //   - bonds neighbor↔outside: neighbor <Sz> fixed, outside <Sz> fixed → cancel ✓
    auto local_mf_energy = [&]() {
        double E = 0;
        for (QClusterMF* qc : touched)
            E += mf_interaction(*qc, qc->eigenstate_idx);
        for (QClusterMF* qc : mf_neighbors)
            E += mf_interaction(*qc, qc->eigenstate_idx);
        return 0.5 * E;
    };

    // Snapshot eigenstate indices for rollback
    struct Snap { QClusterMF* qc; int idx; };
    std::vector<Snap> snaps;
    for (QClusterMF* qc : touched) snaps.push_back({qc, qc->eigenstate_idx});

    double E_local_old = local_mf_energy();

    // Flip ring spins
    for (auto s : p->member_spins) s->ising_val *= -1;

    // Re-diagonalise touched clusters; track to nearest eigenvalue
    for (QClusterMF* qc : touched) {
        double E_old = qc->eigenvalues[qc->eigenstate_idx];
        QClusterMF::BoundaryConfig new_cfg = 0;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i]->ising_val == +1)
                new_cfg |= (1u << i);
        qc->diagonalise(new_cfg);
        int best = 0;
        double best_d = std::abs(qc->eigenvalues[0] - E_old);
        for (int n = 1; n < (int)qc->eigenvalues.size(); n++) {
            double d = std::abs(qc->eigenvalues[n] - E_old);
            if (d < best_d) { best_d = d; best = n; }
        }
        assert(best_d < ENERGY_TOL);
        qc->eigenstate_idx = best;
    }

    // Metropolis on change in local MF energy (ring Ising bonds are zero-energy)
    double dE = local_mf_energy() - E_local_old;
    if (mc.uniform(mc.rng) < std::exp(-mc.beta * dE))
        return 1;

    // Rollback: restore spins and re-diagonalise touched clusters to original config
    for (auto s : p->member_spins) s->ising_val *= -1;
    for (auto& snap : snaps) {
        QClusterMF::BoundaryConfig old_cfg = 0;
        for (int i = 0; i < (int)snap.qc->classical_boundary_spins.size(); i++)
            if (snap.qc->classical_boundary_spins[i]->ising_val == +1)
                old_cfg |= (1u << i);
        snap.qc->diagonalise(old_cfg);  // also restores Sz_expect
        snap.qc->eigenstate_idx = snap.idx;
    }
    return 0;
}



struct MFBond {
    QClusterMF* c1;
    int site1;
    QClusterMF* c2;
    int site2;
};
    

//
// Move 1a: simultaneous flip of all 6 spins on a complete hexagonal plaquette.
// These are zero-energy moves within the classical ice manifold (alternating
// ↑↓↑↓↑↓ pattern has zero net bond energy change on flip).  Essential for
// ergodicity: without ring moves the classical sector can get trapped.
// Always accepted when the plaquette is in the alternating state; rejected otherwise.
int try_flip_ring(Plaq* p, MCSettings& mc) {
    static const double ENERGY_TOL = 1e-9;

    if (!p->is_complete) return 0;

    // Check alternating pattern
    int prev_ising = p->member_spins[0]->ising_val;
    for (int i = 1; i < 6; ++i) {
        if (p->member_spins[i]->ising_val * prev_ising != -1) return 0;
        prev_ising = p->member_spins[i]->ising_val;
    }

    // Collect quantum clusters adjacent to flipped spins
    std::unordered_set<QClusterMF*> touched;
    for (int i = 0; i < 6; i++) {
        Spin* s = p->member_spins[i];
        for (Spin* nb : s->owning_tetras[i%2]->member_spins)
            if (!nb->deleted && nb->is_quantum() && nb->owning_cluster)
                touched.insert(static_cast<QClusterMF*>(nb->owning_cluster));
    }

    // Fast path: no quantum neighbours — ring is purely classical, always accept
    if (touched.empty()) {
        for (auto s : p->member_spins) s->ising_val *= -1;
        return 1;
    }

    // MF neighbors of touched clusters that are not themselves touched.
    // We need their mf_interaction in the local energy sum because their bonds
    // to touched clusters appear in their mf_bonds lists (from the other side).
    std::vector<MFBond> quantum_quantum_bonds; 
    for (QClusterMF* qc : touched) {
        for (const auto& b : qc->mf_bonds) {
            if (!touched.count(b.other)) {
                quantum_quantum_bonds.emplace_back(qc, b.my_site, b.other, b.other_site);
            } else if (b.other < qc){
                // pointer comparison trick avoids double counting
                quantum_quantum_bonds.emplace_back(qc, b.my_site, b.other, b.other_site);
            }
        }
    }

    // sums all de-duplicated bond energies to compare later
    auto local_mf_energy = [&]() {
        double E = 0;
        double Jzz = ModelParams::get().Jzz;
        for (const auto& b : quantum_quantum_bonds) {
            E += Jzz * b.c1->expect_Sz(b.c1->eigenstate_idx, b.site1)
                     * b.c2->expect_Sz(b.c2->eigenstate_idx, b.site2);
        }
        return E;
    };

    // Snapshot eigenstate indices for rollback
    struct Snap { QClusterMF* qc; int idx; };
    std::vector<Snap> snaps;
    for (QClusterMF* qc : touched) snaps.push_back({qc, qc->eigenstate_idx});

    double E_local_old = local_mf_energy();

    // Flip ring spins
    for (auto s : p->member_spins) s->ising_val *= -1;

    // Re-diagonalise touched clusters; track to nearest eigenvalue
    for (QClusterMF* qc : touched) {
        double E_old = qc->eigenvalues[qc->eigenstate_idx];
        QClusterMF::BoundaryConfig new_cfg = 0;
        for (int i = 0; i < (int)qc->classical_boundary_spins.size(); i++)
            if (qc->classical_boundary_spins[i]->ising_val == +1)
                new_cfg |= (1u << i);
        qc->diagonalise(new_cfg);
        int best = 0;
        double best_d = std::abs(qc->eigenvalues[0] - E_old);
        for (int n = 1; n < (int)qc->eigenvalues.size(); n++) {
            double d = std::abs(qc->eigenvalues[n] - E_old);
            if (d < best_d) { best_d = d; best = n; }
        }
        assert(best_d < ENERGY_TOL);
        qc->eigenstate_idx = best;
    }

    // Metropolis on change in local MF energy (ring Ising bonds are zero-energy)
    double dE = local_mf_energy() - E_local_old;
    if (mc.uniform(mc.rng) < std::exp(-mc.beta * dE))
        return 1;

    // Rollback: restore spins and re-diagonalise touched clusters to original config
    for (auto s : p->member_spins) s->ising_val *= -1;
    for (auto& snap : snaps) {
        snap.qc->sync();
        snap.qc->eigenstate_idx = snap.idx;
    }
    return 0;
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
    static const double ENERGY_TOL = 1e-9;

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
    std::unordered_set<QClusterMF*> touched;
    for (Spin* s : path) {
        for (Spin* nb : s->neighbours) {
            if (!nb->deleted && nb->is_quantum() && nb->owning_cluster)
                touched.insert(static_cast<QClusterMF*>(nb->owning_cluster));
        }
    }

    // sums all de-duplicated bond energies to compare later
    std::vector<MFBond> quantum_quantum_bonds;
    for (QClusterMF* qc : touched) {
        for (const auto& b : qc->mf_bonds) {
            if (!touched.count(b.other)) {
                quantum_quantum_bonds.emplace_back(qc, b.my_site, b.other, b.other_site);
            } else if (b.other < qc){
                // pointer comparison trick avoids double counting
                quantum_quantum_bonds.emplace_back(qc, b.my_site, b.other, b.other_site);
            }
        }
    }

    auto local_mf_energy = [&]() {
        double E = 0;
        double Jzz = ModelParams::get().Jzz;
        for (const auto& b : quantum_quantum_bonds) {
            E += Jzz * b.c1->expect_Sz(b.c1->eigenstate_idx, b.site1)
                     * b.c2->expect_Sz(b.c2->eigenstate_idx, b.site2);
        }
        return E;
    };


    // At this point we have not sync()'d the boundary changes 
    // -> spectra still have original values
    double E_local_old = local_mf_energy();


    // Snapshot eigenstate indices for rollback
    struct Snap { QClusterMF* qc; int idx; };
    std::vector<Snap> snaps;
    for (QClusterMF* qc : touched) snaps.push_back({qc, qc->eigenstate_idx});

    for (QClusterMF* qc : touched) {
        const double E_old = qc->energy();
        qc->sync();
        int best_idx = 0;
        double best_dist = std::abs(qc->eigenvalues[0] - E_old);
        for (int n = 1; n < (int)qc->eigenvalues.size(); n++) {
            double d = std::abs(qc->eigenvalues[n] - E_old);
            if (d < best_dist) { best_dist = d; best_idx = n; }
        }
        assert(best_dist<ENERGY_TOL);
        qc->eigenstate_idx = best_idx;
    }

    // Metropolis on change in local MF energy (ring Ising bonds are zero-energy)
    double dE = local_mf_energy() - E_local_old;
    if (mc.uniform(mc.rng) < std::exp(-mc.beta * dE))
        return 1;


    // Rollback: restore spins and re-diagonalise touched clusters to original config
    for (auto s : path) s->ising_val *= -1;
    for (auto& snap : snaps) {
        snap.qc->sync();
        snap.qc->eigenstate_idx = snap.idx;
    }

    return 0;
}

int classical_tetra_charge(const Tetra* t) {
    int q = 0;
    for (Spin* s : t->member_spins) {
        if (s->deleted) continue;
        assert(!s->is_quantum());
        q += s->ising_val;
    }
    return q;
}

// filters out a list of tetras with monopoles on them
std::vector<Tetra*> find_monopole_tetras(const std::vector<Tetra*>& intact_tetras){
    std::vector<Tetra*> retval;
    for (auto t : intact_tetras) {
        if (classical_tetra_charge(t) != 0) { retval.push_back(t); }
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
    int q_tail = classical_tetra_charge(tail_tetra);
    if (q_tail == 0) return 0; // refuse

    // heal_val for the first step matches sign(Q_tail): flipping a spin with
    // ising_val == heal_val moves the charge out of tail_tetra (Q_tail → 0 if |Q|=2)
    // and creates a charge of opposite sign in the next tetrahedron.
    int heal_val = (q_tail > 0) ? +1 : -1; // majority spin type, flippable type

    // Count candidates at the tail before any flips — needed for the Hastings
    // correction.  Detailed balance analysis shows the intermediate step ratios
    // cancel exactly; only the tail (step 0) and head (reverse step 0) counts
    // differ.
    int n_tail_candidates = 0;
    for (Spin* s : tail_tetra->member_spins)
        if (!s->deleted && !s->is_quantum() && s->ising_val == heal_val)
            ++n_tail_candidates;

    std::vector<Spin*> path;
    Tetra* head_tetra = tail_tetra;
    Spin* prev_spin  = nullptr;
    const int target_length = 1+std::poisson_distribution<int>(target_length_mean)(mc.rng);

    while ((int)path.size() < target_length) {
        // nose-to-tail candidates: non-deleted, non-quantum spins in head with
        // ising_val == heal_val (not the spin we just came through)
        std::vector<Spin*> candidates;
        for (Spin* s : head_tetra->member_spins) {
            if (s == prev_spin || s->deleted || s->is_quantum() ) continue;
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

        if (head_tetra == tail_tetra) {
            // closed loop!
            break;
        }
    }

    // backtrack until head contains no quantum spins
    while (path.size() > 0){
        bool head_is_classical=true;
        for (auto s : head_tetra->member_spins){
            if (s->deleted) continue;
            if (s->is_quantum()) {
                head_is_classical=false; break;
            }
        }
        if (head_is_classical) break;

        Spin* s = path.back();
        s->ising_val *= -1;
        head_tetra = s->owning_tetras[0] == head_tetra ?
            s->owning_tetras[1] : s->owning_tetras[0];
        path.pop_back();
    }

    if (path.size() == 0) return 0;
    if (path.size() == 1) {path[0]->ising_val*=-1; return 0;}


    // ΔE is purely from head and tail tetras (all intermediate tetras have no net field change).
    // Using sigma convention: E(tetra) = Jzz/2 * (Q²-4), so ΔE = Jzz/2 * ΔQ²
    const double Jzz = ModelParams::get().Jzz;
    int Q_tail_proposed = classical_tetra_charge(tail_tetra);
    // q_head is current charge at head; undo the last flip to get initial charge
    int Q_head_proposed = classical_tetra_charge(head_tetra);
    int Q_head_initial = Q_head_proposed - 2 * path.back()->ising_val;
    double dE = (Jzz / 2.0) * (
        (double)(Q_tail_proposed * Q_tail_proposed - q_tail * q_tail) +
        (double)(Q_head_proposed * Q_head_proposed - Q_head_initial * Q_head_initial)
    );

    // Hastings correction.  Intermediate tetras are all at ice-rule (Q=0) before
    // being visited, so their two "other" spins always have one +1 and one −1
    // (they sum to zero), giving n_fwd_k = n_rev_k = 2 at every intermediate step.
    // Only the tail (Q≠0, step 0 of forward) and head (step 0 of reverse) differ:
    //
    //   q(x→x') = 1/n_tail,  q(x'→x) = 1/n_head
    //   Hastings = q(x'→x)/q(x→x') = n_tail / n_head
    //
    // In the proposed state path.back() sits in head_tetra with ising_val =
    // reverse_heal_val and is always a reverse candidate, so n_head ≥ 1.
    const int reverse_heal_val = path.back()->ising_val;
    int n_head_candidates = 0;
    for (Spin* s : head_tetra->member_spins)
        if (!s->deleted && !s->is_quantum() && s->ising_val == reverse_heal_val)
            ++n_head_candidates;

    // --- Metropolis-Hastings ---
    const double hastings = (double)n_tail_candidates / (double)n_head_candidates;
    {
        double E_new = mc.muca ? mc.muca->E_current + dE : 0.0;
        double log_acc;
        if (mc.muca == nullptr) {
            log_acc = -mc.beta * dE + std::log(hastings);
        } else {
            int k_old = mc.muca->energy_bin(mc.muca->E_current);
            int k_new = mc.muca->energy_bin(E_new);
            if (k_old < 0 || k_new < 0) { for (Spin* s : path) s->ising_val *= -1; return 0; }
            log_acc = mc.muca->lnG[k_old] - mc.muca->lnG[k_new] + std::log(hastings);
        }
        if (mc.uniform(mc.rng) < std::exp(log_acc)) {
            if (mc.muca) mc.muca->E_current = E_new;
        } else {
            for (Spin* s : path) s->ising_val *= -1;
            return 0;
        }
    }



    // Update boundary configs of any quantum clusters adjacent to flipped spins.
    // The worm is a purely classical move: the quantum cluster should follow
    // adiabatically. 
    //
    // Needs to i) stay on an eigenstate whose energy is equal to its pre-worm
    // energy in the new Hamiltonian, and ii) ensure that <Sz> remains unaffected.
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

        // Find eigenstate in new basis with energy closest to E_old (must be exact match!)
        int best_idx = 0;
        double best_dist = std::abs(qc->eigenvalues[0] - E_old); //+ (qc->get_Sz_expect(0) - Sz_old).norm();
        for (int n = 1; n < (int)qc->eigenvalues.size(); n++) {
            double d = std::abs(qc->eigenvalues[n] - E_old); //+ (qc->get_Sz_expect(n) - Sz_old).norm();
            if (d < best_dist) { best_dist = d; best_idx = n; }
        }
        if (best_dist > 1e-8) {
            std::cerr<<"Original E:"<<E_old;
            std::cerr<<"\nBest E:"<<qc->eigenvalues[best_idx]<<"\n";
            printf("Spectrum:\n");
            for (auto e : qc->eigenvalues){
                printf("%f ",e);
            }
            printf("\nCluster boundaries:\n");
            for (int i=0; i<(int)qc->classical_boundary_spins.size(); i++){
                printf("%d %p\n", i, (void*)qc->classical_boundary_spins[i]);
            }
            printf("Path:\n");
            uint32_t changed_spins=0;
            for (auto s : path){
                int i_match = -1;
                for (int i=0; i<(int)qc->classical_boundary_spins.size() && i_match <0; i++){
                    if(qc->classical_boundary_spins[i] == s) i_match=i;
                }
                if (i_match>=0){
                    changed_spins |= (1<<i_match);
                }
                printf("%p %d sigma=%d\n", (void*)s, i_match, s->ising_val);
            }


            printf("New boundary: %x\n", new_cfg); 


            qc->diagonalise(new_cfg ^ changed_spins);
            printf("Old Spectrum:\n");
            for (auto e : qc->eigenvalues){
                printf("%f ",e);
            }
            throw std::runtime_error("Bad cluster reassignment");
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

    if (mc.muca == nullptr) {
        // Canonical mode: Gibbs (Boltzmann) sampling — avoids self-loop bias at low T.
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
    } else {
        // MUCA mode: uniform proposal + MUCA acceptance.
        // Avoids the Hastings correction that the Gibbs proposal would require.
        int new_idx = std::uniform_int_distribution<int>(0, dim - 1)(mc.rng);
        double dE = (qc.eigenvalues[new_idx] + mf_interaction(qc, new_idx))
                  - (qc.eigenvalues[old_idx] + mf_interaction(qc, old_idx));
        return muca_accept(mc, dE, 1.0, [&]{ qc.eigenstate_idx = new_idx; });
    }
}


// Move 3 (QClusterMF): Metropolis flip of a boundary spin — exact version.
// Re-diagonalises each affected cluster speculatively to get the exact eigenvalue shift.
// On accept: *qc = move(tmp) installs precomputed eigenvalues + Sz_expect atomically.
// Active when MCStateMF::sweep<true> is used; the cheaper MF version is sweep<false>.
int try_flip_boundary_spin_MF_exact(MCSettings& mc, Spin* s) {
    const double Jzz = ModelParams::get().Jzz;
    double dE_classical = -2.0 * classical_bond_energy(s, Jzz);

    struct ClusterUpdate {
        QClusterMF* qc;
        QClusterMF::BoundaryConfig new_config;
        QClusterMF tmp;
    };
    std::vector<ClusterUpdate> updates;
    updates.reserve(6);

    // Pass 1: build all speculative diagonalisations before computing dE.
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
        QClusterMF tmp = *qc;
        tmp.diagonalise(new_config);
        updates.push_back({qc, new_config, std::move(tmp)});
    }

    // Pass 2: accumulate dE_quantum using updated <Sz> values for all touched clusters.
    // When two updated clusters share an MF bond, both sides have changed <Sz>; count
    // the pair once (pointer ordering) using new values on both sides to get the exact
    // cross-term Jzz * (Sz_A_new * Sz_B_new - Sz_A_old * Sz_B_old).
    double dE_quantum = 0.0;
    for (const auto& u : updates) {
        dE_quantum += u.tmp.eigenvalues[u.qc->eigenstate_idx] - u.qc->energy();

        for (const auto& b : u.qc->mf_bonds) {
            auto it = std::find_if(updates.begin(), updates.end(),
                                   [&](const ClusterUpdate& v){ return v.qc == b.other; });
            if (it != updates.end()) {
                if (b.other < u.qc) continue;  // count each inter-update pair once
                dE_quantum += Jzz * (
                    u.tmp.expect_Sz(u.qc->eigenstate_idx, b.my_site)
                        * it->tmp.expect_Sz(b.other->eigenstate_idx, b.other_site)
                  - u.qc->expect_Sz(u.qc->eigenstate_idx, b.my_site)
                        * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site));
            } else {
                dE_quantum += Jzz * (
                    u.tmp.expect_Sz(u.qc->eigenstate_idx, b.my_site)
                        * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site)
                  - u.qc->expect_Sz(u.qc->eigenstate_idx, b.my_site)
                        * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site));
            }
        }
    }

    double dE_total = dE_classical + dE_quantum;
    return muca_accept(mc, dE_total, 1.0, [&]{
        s->ising_val *= -1;
        for (auto& u : updates)
            *u.qc = std::move(u.tmp);
    });
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

    return muca_accept(mc, dE, 1.0, [&]{
        s->ising_val *= -1;
        for (auto& u : updates)
            u.qc->diagonalise(u.new_config);  // O(1) cache lookup; updates eigenvalues + Sz_expect + boundary_config
    });
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



void find_class_tetras(const std::vector<Spin>& spins,
        std::vector<Tetra*>& classical_tetras)
{

    // Build the static list of non-quantum tetrahedra.
    // Q values change during the simulation but intact status does not.
    classical_tetras.resize(0);
    std::unordered_set<Tetra*> seen_tetras;
    for (auto s : spins) {
        for (Tetra* t : s.owning_tetras) {
            if (!t || !seen_tetras.insert(t).second) continue;
            bool all_class = true;
            for (Spin* m : t->member_spins)
                if (m->is_quantum()) { all_class = false; break; }
            if (all_class) classical_tetras.push_back(t);
        }
    }
}


template<bool UseExactBoundary>
void MCStateMF::sweep(MCSettings& mc_) {
    mc_.sweeps_attempted++;
    if (mc_.moves & MOVE_RING) {
        for (auto p : intact_plaqs)
            mc_.accepted_plaq += try_flip_ring(p, mc_);
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
        auto monopoles = find_monopole_tetras(class_tetras);
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

    find_class_tetras(spins, class_tetras);
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
    return classical_energy() + cluster_energy() + cluster_cluster_energy();
}


double MCStateMF::classical_energy() {
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
    return E;
}

double MCStateMF::cluster_energy(){
    // cluster eigenvalues (includes classical and quantum boundary couplings)
    double E=0;
    for (const auto& qc : clusters) {
        E += qc.energy();
    }
    return E;
}

double MCStateMF::cluster_cluster_energy(){
    // MF cross-terms: factor 1/2 to avoid double-counting (each bond in mf_bonds from both sides)
    double E=0;
    double Jzz = ModelParams::get().Jzz;

    for (const auto& qc : clusters) {
        for (const auto& b : qc.mf_bonds) {
            E += 0.5 * Jzz * qc.expect_Sz(qc.eigenstate_idx, b.my_site)
                           * b.other->expect_Sz(b.other->eigenstate_idx, b.other_site);
        }
    }

    return E;
}

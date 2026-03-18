#include "quantum_cluster.hpp"
#include <unordered_set>

// ─────────────────────────────────────────────────────────────
// QClusterBase — shared implementations
// ─────────────────────────────────────────────────────────────

void QClusterBase::build_index_maps() {
    spin_to_idx_.clear();
    spin_to_bdry_idx_.clear();
    for (int i = 0; i < n_spins(); i++)
        spin_to_idx_[spins[i]] = i;
    for (int i = 0; i < (int)boundary_spins.size(); i++)
        spin_to_bdry_idx_[boundary_spins[i]] = i;
}

void QClusterBase::build_sz_cache(const Eigen::MatrixXd& evectors) {
    const int dim = hilbert_dim();
    const int ns  = n_spins();

    // Z(b, j) = z_j(b) ∈ {-1,+1}
    Eigen::MatrixXd Z(dim, ns);
    for (int j = 0; j < ns; j++)
        for (int b = 0; b < dim; b++)
            Z(b, j) = ((b >> j) & 1) ? -1.0 : 1.0;

    // sz_cache_(j, n) = Σ_b Z(b,j) · evectors(b,n)²
    sz_cache_.noalias() = Z.transpose() * evectors.array().square().matrix();
}

double QClusterBase::sz_expectation_at(int site_idx, int n) const {
    return sz_cache_(site_idx, n);
}

// Apply Jzz · dsigma · Σ_{j∈interior_neighbours} z_j(b) to H's diagonal.
// Used to add or remove a single boundary spin's contribution.
static void apply_bdry_delta(Eigen::MatrixXd& H,
                              Spin* bspin, double dsigma, double Jzz,
                              const std::unordered_map<const Spin*, int>& spin_to_idx,
                              int dim)
{
    for (Spin* nb : bspin->neighbours) {
        auto it = spin_to_idx.find(nb);
        if (it == spin_to_idx.end()) continue;
        int site_j = it->second;
        for (int b = 0; b < dim; b++) {
            double zj = ((b >> site_j) & 1) ? -1.0 : 1.0;
            H(b, b) += Jzz * dsigma * zj;
        }
    }
}

void QClusterBase::diagonalise(BoundaryConfig config) {
    double Jzz = ModelParams::get().Jzz;
    const int dim = hilbert_dim();

    // Build H_current_ = H_base_ + all boundary terms
    H_current_ = H_base_;
    for (int i = 0; i < (int)boundary_spins.size(); i++)
        apply_bdry_delta(H_current_, boundary_spins[i], config[i], Jzz, spin_to_idx_, dim);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_current_);
    eigenvalues_     = solver.eigenvalues();
    build_sz_cache(solver.eigenvectors());
    boundary_config_ = std::move(config);
}

// Speculative re-diagonalisation with an arbitrary new boundary-field vector.
// Applies delta from boundary_config_ → new_config to H_current_, diagonalises,
// then undoes the delta so H_current_ is unchanged on return.
QClusterBase::Snapshot QClusterBase::propose_with_config(BoundaryConfig new_config) const {
    double Jzz = ModelParams::get().Jzz;
    const int dim = hilbert_dim();

    // Apply delta (only changed entries)
    for (int i = 0; i < (int)boundary_spins.size(); i++) {
        double dsigma = new_config[i] - boundary_config_[i];
        if (dsigma == 0.0) continue;
        apply_bdry_delta(H_current_, boundary_spins[i], dsigma, Jzz, spin_to_idx_, dim);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_current_);

    // Undo delta
    for (int i = 0; i < (int)boundary_spins.size(); i++) {
        double dsigma = boundary_config_[i] - new_config[i];
        if (dsigma == 0.0) continue;
        apply_bdry_delta(H_current_, boundary_spins[i], dsigma, Jzz, spin_to_idx_, dim);
    }

    Snapshot snap;
    snap.new_config_       = std::move(new_config);
    snap.new_eigenvalues_  = solver.eigenvalues();
    snap.new_eigenvectors_ = solver.eigenvectors();
    return snap;
}

// Flip boundary_config_[bidx] sign in-place on H_current_, diagonalise, undo.
// Only valid for ±1 classical boundary spins.
QClusterBase::Snapshot QClusterBase::propose_boundary_flip(int bidx) const {
    double Jzz    = ModelParams::get().Jzz;
    const int dim = hilbert_dim();
    double sigma  = boundary_config_[bidx];
    double dsigma = -2.0 * sigma;   // sigma → -sigma

    apply_bdry_delta(H_current_, boundary_spins[bidx], dsigma, Jzz, spin_to_idx_, dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_current_);
    apply_bdry_delta(H_current_, boundary_spins[bidx], -dsigma, Jzz, spin_to_idx_, dim);

    Snapshot snap;
    snap.new_config_       = boundary_config_;
    snap.new_config_[bidx] = -sigma;
    snap.new_eigenvalues_  = solver.eigenvalues();
    snap.new_eigenvectors_ = solver.eigenvectors();
    return snap;
}

void QClusterBase::commit(Snapshot&& snap, int new_eigenstate_idx) {
    double Jzz = ModelParams::get().Jzz;
    const int dim = hilbert_dim();

    // Update H_current_ incrementally
    for (int i = 0; i < (int)boundary_spins.size(); i++) {
        double dsigma = snap.new_config_[i] - boundary_config_[i];
        if (dsigma == 0.0) continue;
        apply_bdry_delta(H_current_, boundary_spins[i], dsigma, Jzz, spin_to_idx_, dim);
    }

    eigenvalues_     = std::move(snap.new_eigenvalues_);
    boundary_config_ = std::move(snap.new_config_);
    eigenstate_idx_  = new_eigenstate_idx;
    build_sz_cache(snap.new_eigenvectors_);
}


// ─────────────────────────────────────────────────────────────
// QCluster — exact, classical-only boundary
// ─────────────────────────────────────────────────────────────

void QCluster::detect_boundary_spins() {
    std::unordered_set<Spin*> seen;
    for (auto s : spins) {
        for (auto nb : s->neighbours) {
            if (!nb->deleted && nb->owning_cluster == nullptr && seen.insert(nb).second)
                boundary_spins.push_back(nb);
        }
    }
}

void QCluster::build_matrix_rep_1o() {
     if (spins.size() >MAX_CLUSTER_SPINS)
        throw std::runtime_error("Cluster size exceeds MAX_CLUSTER_SPINS");

    const int dim = hilbert_dim();
    H_base_ = Eigen::MatrixXd::Zero(dim, dim);

    double Jzz = ModelParams::get().Jzz;
    double Jxx = ModelParams::get().Jxx;
    double Jyy = ModelParams::get().Jyy;

    for (int site_i = 0; site_i < n_spins(); site_i++) {
        Spin* si = spins[site_i];
        for (Spin* nb : si->neighbours) {
            int site_j = spin_index(nb);
            if (site_j <= site_i) continue;
            for (int b = 0; b < dim; b++) {
                int zi = ((b >> site_i) & 1) ? -1 : +1;
                int zj = ((b >> site_j) & 1) ? -1 : +1;
                H_base_(b, b) += Jzz * zi * zj;

                // XX + YY only across bonds where both far-end tetras are free
                int end_sl = (si->owning_tetras[0] != nb->owning_tetras[0]) ? 0 : 1;
                if (si->owning_tetras[end_sl]->can_fluctuate_free &&
                    nb->owning_tetras[end_sl]->can_fluctuate_free) {
                    int b_x_flip = b ^ ((1 << site_i) | (1 << site_j));
                    H_base_(b, b_x_flip) += Jxx - Jyy * zi * zj;
                }
            }
        }
    }
}

void QCluster::initialise() {
    build_index_maps();
    BoundaryConfig cfg(boundary_spins.size(), 0.0);
    for (size_t i = 0; i < boundary_spins.size(); i++)
        cfg[i] = (double)boundary_spins[i]->ising_val;
    build_matrix_rep_1o();
    diagonalise(std::move(cfg));
    eigenstate_idx_ = 0;
}


// ─────────────────────────────────────────────────────────────
// QClusterMF — mean-field, quantum+classical boundary
// ─────────────────────────────────────────────────────────────

void QClusterMF::detect_boundary_spins() {
    std::unordered_set<Spin*> seen;
    for (auto s : spins) {
        for (auto nb : s->neighbours) {
            if (!nb->deleted && nb->owning_cluster != this && seen.insert(nb).second)
                boundary_spins.push_back(nb);
        }
    }
}

void QClusterMF::build_matrix_rep() {
    if (spins.size() > MAX_CLUSTER_SPINS)
        throw std::runtime_error("Cluster size exceeds MAX_CLUSTER_SPINS");

    const int dim = hilbert_dim();
    H_base_ = Eigen::MatrixXd::Zero(dim, dim);

    double Jzz = ModelParams::get().Jzz;
    double Jxx = ModelParams::get().Jxx;
    double Jyy = ModelParams::get().Jyy;

    for (int site_i = 0; site_i < n_spins(); site_i++) {
        Spin* si = spins[site_i];
        for (Spin* nb : si->neighbours) {
            int site_j = spin_index(nb);
            if (site_j <= site_i) continue;
            for (int b = 0; b < dim; b++) {
                int zi = ((b >> site_i) & 1) ? -1 : +1;
                int zj = ((b >> site_j) & 1) ? -1 : +1;
                // ZZ for all intra-cluster bonds
                H_base_(b, b) += Jzz * zi * zj;
                // XX + YY only across bonds where both far-end tetras are free
                // (same guard as QCluster — inter-sub-cluster ZZ bonds are handled
                // via MF boundary fields, not here)
                int end_sl = (si->owning_tetras[0] != nb->owning_tetras[0]) ? 0 : 1;
                if (si->owning_tetras[end_sl]->can_fluctuate_free &&
                    nb->owning_tetras[end_sl]->can_fluctuate_free) {
                    int b_x_flip = b ^ ((1 << site_i) | (1 << site_j));
                    H_base_(b, b_x_flip) += Jxx - Jyy * zi * zj;
                }
            }
        }
    }
}

void QClusterMF::initialise() {
    build_index_maps();
    BoundaryConfig cfg(boundary_spins.size(), 0.0);
    for (size_t i = 0; i < boundary_spins.size(); i++) {
        Spin* bs = boundary_spins[i];
        if (bs->owning_cluster && bs->owning_cluster->is_initialized()) {
            int site = bs->owning_cluster->spin_index(bs);
            cfg[i] = bs->owning_cluster->sz_expectation(site);
        } else {
            cfg[i] = (double)bs->ising_val;
        }
    }
    build_matrix_rep();
    diagonalise(std::move(cfg));
    eigenstate_idx_ = 0;
}

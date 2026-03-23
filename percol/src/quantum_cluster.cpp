#include "quantum_cluster.hpp"

// Construct H_base in the 2^N-dimensional Hilbert space of the cluster.
//
// Basis convention: basis state b encodes spin i as bit i of b.
//   bit = 1  →  zi = +1  (spin up)
//   bit = 0  →  zi = −1  (spin down)
//
// For each intra-cluster bond (i < j):
//   Diagonal:     H(b,b)      += Jzz * zi(b) * zj(b) 
//   Off-diagonal: H(b,b_flip) += Jxx − Jyy * zi(b) * zj(b)   [XX−YY spin-flip]
//     where b_flip = b XOR (1<<i | 1<<j)
//     — only on bonds where ends_can_fluctuate(), i.e. both end tetrahedra are defects.
//
// CAUTION: Jxx and Jyy are read from ModelParams here and frozen into H_base.
// Set ModelParams before calling initialise().
void QClusterBase::build_matrix_rep(){
    if (spins.size() > MAX_CLUSTER_SPINS){
        throw std::runtime_error("Cluster encountered exceeding max cluster size");
    }
    const int dim = hilbert_dim();
    H_base = Eigen::MatrixXd::Zero(dim, dim);

    double Jzz = ModelParams::get().Jzz;
    double Jxx = ModelParams::get().Jxx;
    double Jyy = ModelParams::get().Jyy;

    for (int site_i = 0; site_i < n_spins(); site_i++) {
        Spin* si = spins[site_i];
        for (Spin* nb : si->neighbours) {
            int site_j = spin_index(nb);
            if (site_j <= site_i) continue; // each bond once
            // Z_i Z_j is diagonal: basis state |b> has sigma_i = ±1 = 2*bit_i - 1
            for (int b = 0; b < dim; b++) {
                // ZZ terms: + Jzz * Z_i * Z_j for each intra-cluster bond
                int sigma_i = ((b >> site_i) & 1) ? +1 : -1;
                int sigma_j = ((b >> site_j) & 1) ? +1 : -1;
                H_base(b, b) += Jzz * sigma_i * sigma_j ;

                // only add this if both of the tetras are free to oscillate
                if (ends_can_fluctuate(si, nb)){
                    int b_x_flip = b ^ ((1 << site_i) | (1 << site_j));
                    H_base(b, b_x_flip) += (Jxx - Jyy * sigma_i * sigma_j) ;
                }
            }
        }
    }
}

// Read the current ising_val of each classical boundary spin and pack
// them into the boundary_config bitmask (bit i set ↔ ising_val == +1).
void QClusterBase::initialise(){
    boundary_config = static_cast<QClusterBase::BoundaryConfig>(0);
    for (size_t i=0; i< classical_boundary_spins.size(); i++){
        if (classical_boundary_spins[i]->ising_val == 1){
            boundary_config |= (1ull << i);
        }
    }
}

// Build H_base, precompute the spectrum for every possible boundary config
// (2^k entries, k = |classical_boundary_spins|), then set the initial
// eigenvalues via a cache lookup.  The shared_ptr cache is immutable after
// this point, so O(1) shallow copies in try_flip_boundary_spin are safe.
void QCluster::initialise(){
    this->QClusterBase::initialise();
    build_matrix_rep();

    if ((int)classical_boundary_spins.size() <= MAX_CACHED_BOUNDARY) {
        int n_configs = 1 << (int)classical_boundary_spins.size();
        auto cache = std::make_shared<std::vector<Eigen::VectorXd>>(n_configs);
        for (int cfg = 0; cfg < n_configs; cfg++) {
            auto H = ham_with_classical_bcs(classical_boundary_spins, (uint32_t)cfg);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
            (*cache)[cfg] = solver.eigenvalues();
        }
        eval_cache = cache;
    }

    diagonalise(boundary_config);
}

// As QCluster::initialise(), but also precomputes Sz_expect for every
// boundary config (needed by mf_interaction) and builds the mf_bonds list
// of cross-cluster quantum bonds that will be handled via mean field.
void QClusterMF::initialise(){
    this->QClusterBase::initialise();
    build_matrix_rep();

    if ((int)classical_boundary_spins.size() <= MAX_CACHED_BOUNDARY) {
        int n_configs = 1 << (int)classical_boundary_spins.size();
        auto ecache = std::make_shared<std::vector<Eigen::VectorXd>>(n_configs);
        auto scache = std::make_shared<std::vector<Eigen::MatrixXd>>(n_configs);
        for (int cfg = 0; cfg < n_configs; cfg++) {
            auto H = ham_with_classical_bcs(classical_boundary_spins, (uint32_t)cfg);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
            (*ecache)[cfg] = solver.eigenvalues();
            const auto& evals = solver.eigenvalues();
            const auto& psi   = solver.eigenvectors();
            Eigen::MatrixXd sz(evals.size(), (int)spins.size());
            for (int site_i = 0; site_i < (int)spins.size(); site_i++) {
                for (int n = 0; n < (int)evals.size(); n++) {
                    double val = 0;
                    for (int b = 0; b < hilbert_dim(); b++) {
                        double zi = ((b >> site_i) & 1) ? +1.0 : -1.0;
                        val += psi(b, n) * psi(b, n) * zi;
                    }
                    sz(n, site_i) = val;
                }
            }
            (*scache)[cfg] = sz;
        }
        eval_cache = ecache;
        sz_cache   = scache;
    }

    diagonalise(boundary_config);

    // Enumerate cross-cluster bonds: pairs (my quantum spin, their quantum spin)
    // that are nearest neighbours but belong to different clusters.
    // These are not in H_base; their Jzz coupling is approximated via <Sz>.
    mf_bonds.clear();
    for (int i = 0; i < (int)spins.size(); i++) {
        for (Spin* j : quantum_boundary_spins) {
            bool bonded = std::find(spins[i]->neighbours.begin(),
                                    spins[i]->neighbours.end(), j)
                          != spins[i]->neighbours.end();
            if (!bonded) continue;
            auto* nb_qc = static_cast<QClusterMF*>(j->owning_cluster);
            mf_bonds.push_back({i, nb_qc, nb_qc->spin_index(j)});
        }
    }
}

// Add the diagonal boundary-spin coupling to a copy of H_base and return it.
//
// For each classical boundary spin i with value sigma_i = ±1 (from classical_config),
// and each cluster spin j neighbouring it:
//   H(b,b) += Jzz * sigma_i * zj(b)    for all basis states b
//
// Jzz is read from ModelParams at call time (not frozen like Jxx/Jyy).
// This is the only term that changes when a boundary spin flips.
inline Eigen::MatrixXd QClusterBase::ham_with_classical_bcs(const std::vector<Spin*>& classical_spins, uint32_t classical_config){

    Eigen::MatrixXd H = H_base; // copy, then add boundary terms
    double Jzz = ModelParams::get().Jzz;

    // For each boundary spin i, add J * sigma_i * Z_{cluster_site}
    // where sigma_i = ±1 and cluster_site is the cluster spin neighbouring classical_spins[i]
    for (int i = 0; i < (int)classical_spins.size(); i++) {
        int sigma = (classical_config >> i) & 1 ? +1 : -1;
        Spin* bspin = classical_spins[i];
        // find which cluster spin(s) are neighbours of this boundary spin
        for (Spin* nb : bspin->neighbours) {
            int site_j = spin_index(nb);
            if (site_j < 0) continue; // not in cluster
            // add J * sigma * Z_{site_j} (diagonal)
            for (int b = 0; b < hilbert_dim(); b++) {
                int sigma_j = ((b >> site_j) & 1) ? +1 : -1;
                H(b, b) += Jzz * sigma * sigma_j;
            }
        }
    }

    return H;

}

// Update eigenvalues for the given boundary config.
// Fast path: O(1) cache lookup if the cluster has ≤ MAX_CACHED_BOUNDARY boundary spins.
// Slow path: full O(dim^3) diagonalisation (for large clusters or uncached case).
void QCluster::diagonalise(BoundaryConfig config) {

    if (eval_cache) {
        eigenvalues     = (*eval_cache)[config];
        boundary_config = config;
        return;
    }

    auto H = ham_with_classical_bcs(classical_boundary_spins, config);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    eigenvalues  = solver.eigenvalues();
    boundary_config = config;
}

// As QCluster::diagonalise(), but also updates Sz_expect(n, site_i) = <n|Sz_i|n>
// for all eigenstates n and cluster sites i.  These expectation values are
// consumed by mf_interaction() of neighbouring clusters.
//
//   Sz_expect(n, i) = Σ_b  |ψ_n(b)|² * z_i(b)
//
// where z_i(b) = +1 if bit i of b is set (spin up), −1 otherwise.
void QClusterMF::diagonalise(BoundaryConfig config) {

    if (eval_cache) {
        eigenvalues     = (*eval_cache)[config];
        Sz_expect       = (*sz_cache)[config];
        boundary_config = config;
        return;
    }

    auto H = ham_with_classical_bcs(classical_boundary_spins, config);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    eigenvalues  = solver.eigenvalues();
    const auto& psi = solver.eigenvectors();
    boundary_config = config;

    Sz_expect.resize(eigenvalues.size(), (int)spins.size());

    // re-evaluate <Sz> on all sites
    // Sz_expect(n, site_i) = <psi_n | S^z_i | psi_n>
    //   = sum_b psi(b,n)^2 * Z_i(b)   where Z_i(b) = +1 if bit i of b is 0, else -1
    for (int site_i = 0; site_i < (int)spins.size(); site_i++) {
        for (int n = 0; n < (int)eigenvalues.size(); n++) {
            double val = 0;
            for (int b = 0; b < hilbert_dim(); b++) {
                double zi = ((b >> site_i) & 1) ? +1.0 : -1.0;
                val += psi(b, n) * psi(b, n) * zi;
            }
            Sz_expect(n, site_i) = val;
        }
    }
}

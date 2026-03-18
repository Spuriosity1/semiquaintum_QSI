#include "quantum_cluster.hpp"

void QClusterBase::build_matrix_rep(){
    if (spins.size() > MAX_CLUSTER_SPINS){
        throw std::runtime_error("Cluster encountered exceeding max cluster size");
    }
    const int dim = hilbert_dim();
    H_base = Eigen::MatrixXd(dim, dim);

    double Jzz = ModelParams::get().Jzz;
    double Jxx = ModelParams::get().Jxx;
    double Jyy = ModelParams::get().Jyy;
    
    for (int site_i = 0; site_i < n_spins(); site_i++) {
        Spin* si = spins[site_i];
        for (Spin* nb : si->neighbours) {
            int site_j = spin_index(nb);
            if (site_j <= site_i) continue; // each bond once
            // Z_i Z_j is diagonal: basis state |b> has Z_i = ±1 = 1 - 2*bit_i
            for (int b = 0; b < dim; b++) {
                // ZZ terms: -Jzz * Z_i * Z_j for each intra-cluster bond
                int zi = ((b >> site_i) & 1) ? -1 : +1;
                int zj = ((b >> site_j) & 1) ? -1 : +1;
                H_base(b, b) += Jzz * zi * zj;

                // only add this if both of the tetras are free to oscillate
                if (ends_can_fluctuate(si, nb)){
                    int b_x_flip = b ^ ((1 << site_i) | (1 << site_j));
                    H_base(b, b_x_flip) += Jxx - Jyy * zi * zj;
                }
            }
        }
    }

}

void QClusterBase::initialise(){
    boundary_config = static_cast<QClusterBase::BoundaryConfig>(0);
    for (size_t i=0; i< classical_boundary_spins.size(); i++){
        if (classical_boundary_spins[i]->ising_val == 1){
            boundary_config |= (1ull << i);
        }
    }
}

void QCluster::initialise(){
    this->QClusterBase::initialise();
    build_matrix_rep();
    diagonalise(boundary_config);
}

void QClusterMF::initialise(){
    this->QClusterBase::initialise();
    build_matrix_rep();
    diagonalise(boundary_config);

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

// private helper
inline Eigen::MatrixXd QClusterBase::diagonalise_ham_classical_bcs(const std::vector<Spin*>& classical_spins, uint32_t classical_config){

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
            // add -J * sigma * Z_{site_j} (diagonal)
            for (int b = 0; b < hilbert_dim(); b++) {
                int zj = ((b >> site_j) & 1) ? -1 : +1;
                H(b, b) += Jzz * sigma * zj;
            }
        }
    }

    return H;

}

  // --- build full H for a given boundary config and diagonalise ---
void QCluster::diagonalise(BoundaryConfig config) {

    auto H = diagonalise_ham_classical_bcs(classical_boundary_spins, config);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    eigenvalues  = solver.eigenvalues();
    boundary_config = config;
}

void QClusterMF::diagonalise(BoundaryConfig config) {

    auto H = diagonalise_ham_classical_bcs(classical_boundary_spins, config);
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
                double zi = ((b >> site_i) & 1) ? -1.0 : +1.0;
                val += psi(b, n) * psi(b, n) * zi;
            }
            Sz_expect(n, site_i) = val;
        }
    }
}

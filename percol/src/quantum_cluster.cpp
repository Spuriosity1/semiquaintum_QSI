#include "quantum_cluster.hpp"

void QCluster::build_matrix_rep(){
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

                int b_x_flip = b ^ ((1 << site_i) | (1 << site_j));
                H_base(b, b_x_flip) += Jxx - Jyy * zi * zj;
            }
        }
    }

}

  // --- build full H for a given boundary config and diagonalise ---
void QCluster::diagonalise(BoundaryConfig config) {
    Eigen::MatrixXd H = H_base; // copy, then add boundary terms

    double Jzz = ModelParams::get().Jzz;

    // For each boundary spin i, add J * sigma_i * Z_{cluster_site}
    // where sigma_i = ±1 and cluster_site is the cluster spin neighbouring boundary_spins[i]
    for (int i = 0; i < (int)boundary_spins.size(); i++) {
        int sigma = (config >> i) & 1 ? +1 : -1;
        Spin* bspin = boundary_spins[i];
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

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    eigenvalues  = solver.eigenvalues();
    eigenvectors = solver.eigenvectors();
    boundary_config = config;
}

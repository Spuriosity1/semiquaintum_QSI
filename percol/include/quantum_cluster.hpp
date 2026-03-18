#pragma once


#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include "lattice_lib/unitcellspec.hpp"
#include <Eigen/Dense>
#include <unordered_map>

struct ModelParams {
    double Jzz = 1.0;
    double Jxx = 0.5;
    double Jyy = 0.5;
//    double h = 0;

    static ModelParams& get() {
        static ModelParams instance;
        return instance;
    }
private:
    ModelParams() = default;
};


struct QClusterBase;
struct QCluster;
struct QClusterMF;
struct Tetra;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;
    std::array<Tetra*, 2> owning_tetras; // sl index
    bool deleted=false;
    Spin* q_cluster_root=nullptr; // null = not quantum
    QClusterBase* owning_cluster=nullptr; // null = classical

    bool is_quantum() const {
        return q_cluster_root != nullptr;
    }

    void reset(){
        q_cluster_root=nullptr;
        owning_cluster=nullptr;
    }

    // state for Monte Carlo
    int ising_val = 1; // all up, why not

    // constructors
    Spin() : ipos(0,0,0){ }
    Spin(const ipos_t& x) : ipos(x) {}
};

struct Tetra {
    // static data
    ipos_t ipos;
    std::vector<Spin*> member_spins;

    struct Neighbour {
        Tetra* tet;
        Spin* via;
    };

    // dynamically updated at runtime
    std::vector<Neighbour> neighbours;
    bool can_fluctuate_free;


    // constructors
    Tetra() : ipos(0,0,0) { }
    Tetra(const ipos_t& x) : ipos(x) {}
};


// ─────────────────────────────────────────────────────────────
// Abstract base: shared storage + all concrete MC-facing methods
// ─────────────────────────────────────────────────────────────
struct QClusterBase {
    using BoundaryConfig = std::vector<double>;

    struct Snapshot {
        double energy_at(int idx) const { return new_eigenvalues_[idx]; }

        double partition_function(double beta) const {
            double Z = 0;
            for (auto& E : new_eigenvalues_) Z += std::exp(-beta * E);
            return Z;
        }

        // r uniform in [0,1). Returns sampled eigenstate index.
        int sample_boltzmann(double beta, double r) const {
            double threshold = r * partition_function(beta);
            double acc = 0;
            int idx = 0;
            for (; idx < (int)new_eigenvalues_.size() - 1; idx++) {
                acc += std::exp(-beta * new_eigenvalues_[idx]);
                if (acc >= threshold) break;
            }
            return idx;
        }

    private:
        friend struct QClusterBase;
        BoundaryConfig  new_config_;
        Eigen::VectorXd new_eigenvalues_;
        Eigen::MatrixXd new_eigenvectors_;
    };

    // ── common data ──────────────────────────────────────────
    std::vector<Spin*> spins;
    std::vector<Spin*> boundary_spins;

    int n_spins()     const { return (int)spins.size(); }
    int hilbert_dim() const { return 1 << n_spins(); }

    // O(1) index lookups (hash maps built at initialise() time)
    int spin_index(const Spin* s) const {
        auto it = spin_to_idx_.find(s);
        return it != spin_to_idx_.end() ? it->second : -1;
    }

    int boundary_index(const Spin* s) const {
        auto it = spin_to_bdry_idx_.find(s);
        return it != spin_to_bdry_idx_.end() ? it->second : -1;
    }

    // True once diagonalise() has been called at least once
    bool is_initialized() const { return eigenvalues_.size() > 0; }

    // ── concrete MC-facing methods ────────────────────────────
    double energy()             const { return eigenvalues_[eigenstate_idx_]; }
    double eigenvalue(int idx)  const { return eigenvalues_[idx]; }
    int    eigenstate_idx()     const { return eigenstate_idx_; }
    void   set_eigenstate(int idx)    { eigenstate_idx_ = idx; }

    // O(1) lookup (cache built after each diagonalisation)
    double sz_expectation_at(int site_idx, int n) const;
    double sz_expectation(int site_idx) const {
        return sz_expectation_at(site_idx, eigenstate_idx_);
    }

    BoundaryConfig boundary_config() const { return boundary_config_; }

    double partition_function(double beta) const {
        double Z = 0;
        for (auto& E : eigenvalues_) Z += std::exp(-beta * E);
        return Z;
    }

    // In-place incremental update of H_current_ — no matrix copy.
    // Flips boundary_config_[bidx] sign (±1 for classical spins).
    Snapshot propose_boundary_flip(int bidx) const;

    // Speculative re-diagonalisation with an arbitrary new boundary-field vector.
    // Computes delta from boundary_config_ and applies only changed terms.
    Snapshot propose_with_config(BoundaryConfig new_config) const;

    // Commit a snapshot: update H_current_, boundary_config_, eigenvalues/vectors,
    // eigenstate_idx_, and rebuild sz_cache_.
    void commit(Snapshot&& snap, int new_eigenstate_idx);

    // ── pure-virtual: subtype-specific behaviour ─────────────
    virtual void detect_boundary_spins() = 0;
    virtual void initialise() = 0;

    virtual ~QClusterBase() = default;

protected:
    Eigen::MatrixXd H_base_;
    BoundaryConfig  boundary_config_;
    Eigen::VectorXd eigenvalues_;

    int             eigenstate_idx_ = 0;

    // Precomputed caches — built by build_boundary_diag() / build_sz_cache()
    // sz_cache_(site, n) = ⟨n|Sz_site|n⟩
    Eigen::MatrixXd sz_cache_;

    // H_current_ = H_base_ + current boundary terms (maintained incrementally)
    mutable Eigen::MatrixXd H_current_;

    // O(1) lookup maps — built at initialise() time
    std::unordered_map<const Spin*, int> spin_to_idx_;
    std::unordered_map<const Spin*, int> spin_to_bdry_idx_;

    // Build index hash maps (call once, at the start of initialise()).
    void build_index_maps();
    // Build boundary_diag_ (call after H_base_ and index maps are ready).
//    void build_boundary_diag();
    // Build sz_cache_ from current eigenvectors_ (call after each diagonalise).
    void build_sz_cache(const Eigen::MatrixXd& eigenvectors);
    // Solve H_base_ + boundary terms for the given config; maintain H_current_.
    void diagonalise(BoundaryConfig config);
};


// ─────────────────────────────────────────────────────────────
// QCluster — exact diagonalisation
// Boundary: classical spins only.  H built with can_fluctuate_free guard.
// ─────────────────────────────────────────────────────────────
struct QCluster : QClusterBase {
    void detect_boundary_spins() override;
    void initialise() override;

private:
    void build_matrix_rep_1o();
};


// ─────────────────────────────────────────────────────────────
// QClusterMF — mean-field (continuous boundary fields)
// Boundary: classical AND quantum-cluster spins of other clusters.
// H built without can_fluctuate_free guard.
// ─────────────────────────────────────────────────────────────
struct QClusterMF : QClusterBase {
    void detect_boundary_spins() override;
    void initialise() override;

private:
    void build_matrix_rep();
};


const static int MAX_CLUSTER_SPINS = 16;

using MyCell = UnitCellSpecifier<Spin, Tetra>;
using SuperLat = Supercell<Spin, Tetra>;

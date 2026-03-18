#pragma once



#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include "lattice_lib/unitcellspec.hpp"
#include <Eigen/Dense>

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
struct Tetra;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;
    std::array<Tetra*, 2> owning_tetras; // sl index
    bool deleted=false;
    Spin* q_cluster_root=nullptr; // null = not quantum
    QClusterBase* owning_cluster=nullptr;


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

struct Plaq {
    ipos_t ipos;
    std::array<Spin*, 6> member_spins; // 6 spins in canonical order
    bool is_complete;

    Plaq() : ipos(0,0,0) {}
    Plaq(const ipos_t& x) : ipos(x) {}
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
    bool can_fluctuate;


    // constructors
    Tetra() : ipos(0,0,0) { }
    Tetra(const ipos_t& x) : ipos(x) {}
};



inline bool ends_can_fluctuate(const Spin* a, const Spin* b){
    int end_sl = (a->owning_tetras[0] == b->owning_tetras[0]) ? 1 : 0; 
    // sl of the tetrahedra on the edges of the bond
    assert(a->owning_tetras[!end_sl] == b->owning_tetras[!end_sl]);
    return a->owning_tetras[end_sl]->can_fluctuate && b->owning_tetras[end_sl]->can_fluctuate;
}


struct QClusterBase {
    using BoundaryConfig = uint32_t; // bitmask over boundary spins

    std::vector<Spin*> spins;
    std::vector<Spin*> classical_boundary_spins;


    // -- current state for MC --
    int eigenstate_idx;

    int n_spins()    const { return (int)spins.size(); }
    int hilbert_dim() const { return 1 << n_spins(); }

    // --- Base Hamiltonian (does not include boundary spins) ---
    Eigen::MatrixXd H_base;

    // --- current spectrum (recomputed whenever boundary_config changes) ---
    Eigen::VectorXd eigenvalues;        // sorted ascending

    // +1 or -1 for boundary spin i given current config
    int boundary_val(int i) const {
        return (boundary_config >> i) & 1 ? +1 : -1;
    }

    // map spin pointer to its index in spins[]
    int spin_index(const Spin* s) const {
        for (int i = 0; i < (int)spins.size(); i++)
            if (spins[i] == s) return i;
        return -1;
    }

    BoundaryConfig boundary_config = 0; // bit i = spin of boundary_spins[i] (+1 if set)

    double energy() const {
        return eigenvalues[eigenstate_idx];
    }

    double partition_function(double beta) const {
        double Z=0;
        for (auto& E : eigenvalues){
            Z += exp(-beta * E);
        }
        return Z;
    }

    // --- build H_base: ZZ bonds + transverse field, no boundary ---
    void build_matrix_rep();
    virtual void diagonalise(BoundaryConfig config) = 0;

    virtual void initialise();

    protected:

    Eigen::MatrixXd diagonalise_ham_classical_bcs(const std::vector<Spin*>& classical_spins, uint32_t classical_config);
};

struct QCluster : public QClusterBase {
    void initialise() override;
    void diagonalise(BoundaryConfig config) override;
};


// boundary_config is treated as boundary conditions of the neighbouring classical spins.
// Quantum spins are treated differently: the mean field <Sz>
struct QClusterMF : public QClusterBase {

    std::vector<Spin*> quantum_boundary_spins;

    struct MFBond { int my_site; QClusterMF* other; int other_site; };
    std::vector<MFBond> mf_bonds;  // precomputed inter-cluster bonded pairs

    void initialise() override;
    void diagonalise(BoundaryConfig config) override;

    double expect_Sz(const Spin* s) const {
        auto i = this->spin_index(s);
        assert(i>=0);
        return Sz_expect(this->eigenstate_idx, i);
    }

    double expect_Sz(int n, int site_i) const {
        return Sz_expect(n, site_i);
    }

    protected:

    Eigen::MatrixXd Sz_expect; // Sz_expect(n, qspin_id) is <Sz> for eigenstate n
                               // qspin_id is one of the spins I own

};

const static int MAX_CLUSTER_SPINS = 16;

using MyCell = UnitCellSpecifier<Spin, Tetra, Plaq>;
using SuperLat = Supercell<Spin, Tetra, Plaq>;

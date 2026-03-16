#pragma once

#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcellspec.hpp"
#include <random>
#include "geometry.hpp"


template<typename T>
concept UFElement = requires(T t) {
    // Actual position
    { t.ipos } -> std::convertible_to<const ipos_t>;
    // list of neighbours for UF purposes
    { t.neighbours } -> std::same_as<std::vector<T*>&>;
    { t.deleted } -> std::convertible_to<bool>;
    // position of element relative to root, summed along path
    { t.dx } -> std::convertible_to<const ipos_t>;
    { t.parent } -> std::convertible_to<T*>;
};


// forward declarations
struct Spin;
struct Plaq;
struct Bond;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;
    std::vector<Plaq*> plaqs_containing_me;
    std::vector<Bond*> bonds_containing_me;
    bool deleted=false;

    // auxiliary
    ipos_t dx;
    Spin* parent;

    Spin() : ipos(0,0,0), parent(this){ }
    Spin(const ipos_t& x) : ipos(x), parent(this) {}
};

struct Bond {
    ipos_t ipos;
    std::array<Spin*, 2> spin_members = {nullptr, nullptr};
    bool deleted = false;

    Bond() : ipos(0,0,0) {}
    Bond(const ipos_t& x) : ipos(x) {}
};

struct Plaq { 
    ipos_t ipos;
    std::vector<Spin*> spin_members;
    bool deleted = false;

    Plaq() : ipos(0,0,0){}
    Plaq(const ipos_t& x) : ipos(x) {}
};


inline ipos_t floordiv(const ipos_t& x, int base){
    return ipos_t(x[0]/base, x[1]/base, x[2]/base);
}

using MyCell = UnitCellSpecifier<Spin, Bond, Plaq>;
using SuperLat = Supercell<Spin, Bond, Plaq>;

// Generates a supercell of cubic dimension L
inline auto initialise_lattice(int L)
{
    using namespace pyrochlore;

    MyCell cell(imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8}));

    for (int fcc_i=0; fcc_i<4; fcc_i++){
        const auto& r0 = fcc[fcc_i];

        for (int mu=0; mu<4; mu++){
            cell.add(Spin(r0 + pyro[mu]));
            cell.add(Plaq(r0 + ipos_t{2,2,2} - pyro[mu]));
        }
        
        for (int mu=0; mu<4; mu++){
            for (int nu=mu+1; nu<4; nu++){
                cell.add(Bond(r0 + floordiv(pyro[mu] + pyro[nu], 2)));
            }
        }
    }

    std::vector<std::pair<int, int>> munu_map;
    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            munu_map.push_back(std::make_pair(mu, nu));
        }
    }

    // The size of the supercell
//    imat33_t Z = imat33_t::from_cols({-L,L,L},{L,-L,L},{L,L,-L});
    auto Z = imat33_t::from_cols({L,0,0}, {0, L, 0}, {0, 0, L});

    Supercell sc = build_supercell<Spin, Bond, Plaq>(cell, Z);



    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [spin_sl, spin] : cell.enumerate_objects<Spin>()){
            // link up first neighbours
            for (const auto& dx : nn_vectors[spin_sl%4]){
                auto r_nn = spin->ipos + dx;
                auto nn = sc.get_object_at<Spin>(r_nn);
                assert_position(nn, r_nn);
                spin->neighbours.push_back(nn);
            }
        }
    
        for (auto [plaq_sl, plaq] : cell.enumerate_objects<Plaq>()){
            // link up the spin neighbours of the plaquettes
            for (const auto& dp : plaq_boundaries[plaq_sl%4]){
                ipos_t R = plaq->ipos + dp;
                Spin* s0 = sc.get_object_at<Spin>(R);
                assert_position(s0, R);
                plaq->spin_members.push_back(s0);
                s0->plaqs_containing_me.push_back(plaq);
            }
        }

        for (auto [bond_sl, bond] : cell.enumerate_objects<Bond>()){
            auto [mu, nu] = munu_map[bond_sl%6];

            auto delta = floordiv(pyro[mu]-pyro[nu], 2);

            auto r0 = bond->ipos + delta;
            auto r1 = bond->ipos - delta;

            Spin* s0 = sc.get_object_at<Spin>(r0);
            Spin* s1 = sc.get_object_at<Spin>(r1);

            assert_position(s0, r0);
            assert_position(s1, r1);

            bond->spin_members[0] = s0;
            bond->spin_members[1] = s1;

            s0->bonds_containing_me.push_back(bond);
            s1->bonds_containing_me.push_back(bond);
            
        }
    }

    return sc;
}


// Marks all neighbours of spin 's' as deleted if any of them have a hole.
inline void set_spin_deleted(Spin& s, bool deleted){
    s.deleted = deleted;
    for (auto b : s.bonds_containing_me){
        b->deleted = b->spin_members[0]->deleted || b->spin_members[1]->deleted;
    }
    for (auto p : s.plaqs_containing_me){
        p->deleted = false;
        for (auto s2 : p->spin_members){
            p->deleted |= s2->deleted;
        }
    }
}

// Quicker version assuming all previous states were correct.
inline void set_spin_deleted_fast(Spin& s, bool deleted){
    if (deleted == s.deleted) return;
    if(deleted == true){
        // viral spread
        for (auto b: s.bonds_containing_me) b->deleted = true;
        for (auto p: s.plaqs_containing_me) p->deleted = true;
    } else {
        // spin restoration, full check necessary
        set_spin_deleted(s, deleted);
    }
}

// Sweeps through all spins, and deletes with probability p
// i.e. p = 0 means 100% clean
inline void spin_sweep(SuperLat& sc, double p, std::mt19937& rng){
    static auto rand01 = std::uniform_real_distribution();

    for (auto& s : sc.get_objects<Spin>()){
        set_spin_deleted(s, rand01(rng) < p);
    }
}



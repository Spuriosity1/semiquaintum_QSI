#pragma once

#include "quantum_cluster.hpp"
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <vector>
#include "geometry.hpp"

inline void output_cluster_hist(std::ostream& os, const std::map<size_t, size_t>& hist, size_t denom){
    if (denom == 1){
        os<<"Clust. Size\tCount\n";
    } else {
        os<<"Clust. Size\tNumber per site\n";
    }
    for (auto& [s, count] : hist){
        os<<s<<"\t"<<1.0*count/denom<<"\n";
    }

}

template<typename ClusterT>
requires std::derived_from<ClusterT, QClusterBase>
inline void output_cluster_dist(std::ostream& os, std::vector<ClusterT>& clusters, size_t denom){
    std::map<size_t, size_t> hist;

    for (auto Q : clusters){
        int size = Q.spins.size();
        hist[size]++;
    }

    output_cluster_hist(os, hist, denom);


}


// N.B. I thiught of optimising this slightly by modifying the deleted spins' 
// neighbours directly, but you pay a linear cost anyway 
inline auto delete_spins(std::mt19937& rng, SuperLat& sc, double p,
    std::unordered_set<Tetra*>& seed_tetras, std::vector<Plaq*>* intact_plaqs=nullptr){
    seed_tetras.clear();
    static auto rand01 = std::uniform_real_distribution();

    for (auto& s : sc.get_objects<Spin>()){
        s.reset();
        if (rand01(rng) < p) {
            s.deleted = true;
        } else {
            s.deleted = false;
        }
    }

    // pass 2: build the tetra connectivity graph and incomplete hexas
    for (const auto& [I, cell] : sc.enumerate_cells() ) {
        for (auto [tetra_sl, t] : cell.enumerate_objects<Tetra>()){
            t->neighbours.clear();
            for (auto& s : t->member_spins){
                if (!s->deleted) {
                    Tetra* t2=s->owning_tetras[(tetra_sl+1)%2];
                    t->neighbours.push_back({t2, s});
                }
            }

            // mark 1-spin and 3-spin as seed tetras
            if (t->neighbours.size() == 3 || t->neighbours.size() == 1){
                seed_tetras.insert(t);
                t->can_fluctuate=true;
            } else {
                t->can_fluctuate=false;
            }

        }

        if (intact_plaqs != nullptr){
            intact_plaqs->resize(0);
            for (auto [plaq_sl, p] : cell.enumerate_objects<Plaq>()){
                p->is_complete=true;
                for (auto& s: p->member_spins){
                    if (s->deleted || s->is_quantum()) {
                        p->is_complete = false;
                        break;
                    }
                }
                if (p->is_complete) intact_plaqs->push_back(p);
            }
        }
    }

    

}




// Generates a supercell of cubic dimension L
inline auto initialise_lattice(int L)
{
    using namespace pyrochlore;

    MyCell cell(imat33_t::from_cols({8,0,0},{0,8,0},{0,0,8}));

    for (int fcc_i=0; fcc_i<4; fcc_i++){
        const auto& r0 = fcc[fcc_i];
        cell.add(Tetra(r0)); // the up tetra
        cell.add(Tetra(r0-ipos_t{2,2,2})); // the down tetra
        // the plaqs
        for (int mu=0; mu<4; mu++){
            cell.add(Spin(r0 + pyro[mu]));
            cell.add(Plaq(r0 + ipos_t{2,2,2} - pyro[mu]));
        }
    }

    std::vector<std::pair<int, int>> munu_map;
    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            munu_map.push_back(std::make_pair(mu, nu));
        }
    }

    auto Z = imat33_t::from_cols({L,0,0}, {0, L, 0}, {0, 0, L});

    Supercell sc = build_supercell<Spin,Tetra,Plaq>(cell, Z);


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

        for (auto [tetra_sl, tetra] : cell.enumerate_objects<Tetra>()){
            tetra->member_spins.clear();

            int eta = 1-2*(tetra_sl % 2);
            for (int mu=0; mu<4; mu++){
                auto r_spin = tetra->ipos + eta*pyro[mu];
                auto spin_n = sc.get_object_at<Spin>(r_spin);
                assert_position(spin_n, r_spin);
                tetra->member_spins.push_back(spin_n);

                spin_n->owning_tetras[tetra_sl%2] = tetra;
            }

            // note that "neighbours" is set later, once the deletion pattern is known
        }

        // thin ring structs for efficient classical MC
        for (auto [plaq_sl, plaq] : cell.enumerate_objects<Plaq>()){
            for (int i=0; i<6; i++ ){
                const auto& dp = plaq_boundaries[plaq_sl%4][i];
                ipos_t R = plaq->ipos + dp;
                Spin* s0 = sc.get_object_at<Spin>(R);
                assert_position(s0, R);
                plaq->member_spins[i] = s0;
            }
        }
    
    }

    return sc;
}



inline Spin* find_q_root(Spin* s){
    Spin* tmp = s;
    while(tmp != tmp->q_cluster_root){
#ifndef NDEBUG
        if (!tmp->is_quantum()) 
            throw std::runtime_error("Non-quantum spin found in union find");
#endif

        tmp = tmp->q_cluster_root;
    }
    return tmp;
}


namespace QuantumRule {


// ─── shared primitive ────────────────────────────────────────────────────────

/*
 * Recursive DFS over the diamond lattice. Marks spins collected along the
 * current path if the endpoint is a defect tetra and depth is within [min, max].
 *
 *   current  : tetra we just arrived at
 *   prev     : tetra we came from (backtrack guard)
 *   seed_tetras : the defect set
 *   path_spins  : spins accumulated so far on this path
 *   depth    : current hop count (1-indexed on arrival)
 *   min_depth: minimum depth at which a defect endpoint counts
 *   max_depth: maximum hop depth to explore
 *   mark_quantum: callable (Spin*) -> void
 */
template <typename MarkFn>
void dfs_mark(
        Tetra* current, Tetra* prev,
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& path_spins,
        int depth, int min_depth, int max_depth,
        MarkFn& mark_quantum)
{
    // check for a valid endpoint
    if (depth >= min_depth && seed_tetras.count(current)) {
        for (Spin* s : path_spins) mark_quantum(s);
    }

    if (depth == max_depth) return;

    for (auto& [next, bond] : current->neighbours) {
        if (next == prev) continue; // no immediate backtrack
        path_spins.push_back(bond);
        dfs_mark(next, current, seed_tetras, path_spins,
                 depth + 1, min_depth, max_depth, mark_quantum);
        path_spins.pop_back();
    }
}

/*
 * Launches DFS from every defect tetra. The start tetra is excluded as an
 * endpoint (a zero-length self-loop is not a path), hence min_depth >= 1.
 */
inline void mark_q_spins_depth(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins,
        int min_depth, int max_depth)
{
    auto mark_quantum = [&](Spin* s) {
        if (!s->is_quantum()) {
            s->q_cluster_root = s;
            quantum_spins.push_back(s);
        }
    };

    for (Tetra* defect_a : seed_tetras) {
        std::vector<Spin*> path_spins;
        path_spins.reserve(max_depth);
        dfs_mark(defect_a, nullptr, seed_tetras, path_spins,
                 0, min_depth, max_depth, mark_quantum);
    }
}

// ─── six marker functions ─────────────────────────────────────────────────────

// length == 1 exactly
inline void eq1nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{ mark_q_spins_depth(seed_tetras, quantum_spins, 1, 1); }

// length <= 2
inline void le2nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{ mark_q_spins_depth(seed_tetras, quantum_spins, 1, 2); }

// length == 2 exactly  (original mark_2nn_q_spins)
inline void eq2nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{ mark_q_spins_depth(seed_tetras, quantum_spins, 2, 2); }

// length <= 3
inline void le3nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{ mark_q_spins_depth(seed_tetras, quantum_spins, 1, 3); }

// length == 2 or 4 exactly  (original mark_4nn_q_spins)
inline void eq24nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{
    mark_q_spins_depth(seed_tetras, quantum_spins, 2, 2);
    mark_q_spins_depth(seed_tetras, quantum_spins, 4, 4);
}

// length <= 4
inline void le4nn(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<Spin*>& quantum_spins)
{ mark_q_spins_depth(seed_tetras, quantum_spins, 1, 4); }


};

using QMarkFn = void(*)(const std::unordered_set<Tetra*>&, std::vector<Spin*>&);

/*
 * 'greedy' search for spins satisfying certain conditions on being near dimers.
 * These are spins present in the motif DEFECT-spin1-spin2-DEFECT.
 * Needs to be efficient only in the dilute case; we simply consider the full n^2 
 *
 */
template <QMarkFn MarkFn>
inline void identify_quantum_clusters(
        const std::unordered_set<Tetra*>& seed_tetras, 
        std::vector<QCluster>& clust
        ){

    std::vector<Spin*> quantum_spins;

    MarkFn(seed_tetras, quantum_spins);

    // union find
    for (auto qs : quantum_spins){
        for (auto ns : qs->neighbours){
            if (ns->q_cluster_root != nullptr){ // ns is a quantum spin
                // Union: point ns's root to qs's root
                Spin* root_qs = find_q_root(qs);
                Spin* root_ns = find_q_root(ns);
                if (root_qs != root_ns){
                    root_ns->q_cluster_root = root_qs;
                }
            }
        }
    }

    
    // Group spins by their root into QClusters
    std::unordered_map<Spin*, QCluster> root_to_cluster;
    for (auto qs : quantum_spins){
        Spin* root = find_q_root(qs);
        root_to_cluster[root].spins.push_back(qs);
    }

    clust.clear();
    clust.reserve(root_to_cluster.size());
    for (auto& [root, cluster] : root_to_cluster){
        cluster.spins.shrink_to_fit();
        clust.push_back(std::move(cluster));
    }

    // link up the spins' "owning_cluster" pointers
    for (auto& qc : clust){
        for (auto& s : qc.spins){
            s->owning_cluster = &qc;
        }
    }

    // detect and store the boundary spins
    for (auto& qc : clust){
        std::unordered_set<Spin*> seen;
        for (auto& s : qc.spins){
            for (const auto nb : s->neighbours){
                if (!nb->deleted 
                        && nb->owning_cluster == nullptr
                        && seen.insert(nb).second) 
                    qc.classical_boundary_spins.push_back(nb);
            }
        }
    }
}




// More specialised than the above method. We consider neighbouring spins to be
// part of the same cluster only if two of the diamond walks overlap 
// (i.e. clusters end at defect tetras).
inline void identify_1o_clusters(
        const std::unordered_set<Tetra*>& seed_tetras,
        std::vector<QClusterMF>& clust){

    // Simpler algorithm. Walk all unique bonds, and check whether
    // the terminating tetras are quantum

    std::vector<Spin*> quantum_spins;
    QuantumRule::eq2nn(seed_tetras, quantum_spins); // colour spins by distance

    // union find
    for (auto qs : quantum_spins){
        for (auto ns : qs->neighbours){
            if (ns->q_cluster_root != nullptr &&
                    ns < qs &&
                    ends_can_fluctuate(ns, qs) ) {
                // avoids double counting
                // mark as quantum
                Spin* root_qs = find_q_root(qs);
                Spin* root_ns = find_q_root(ns);
                if (root_qs != root_ns){
                    root_ns->q_cluster_root = root_qs;
                }
            }   
        }
    }

    // Group spins by their root into QClusterMFs
    std::unordered_map<Spin*, QClusterMF> root_to_cluster;
    for (auto qs : quantum_spins){
        Spin* root = find_q_root(qs);
        root_to_cluster[root].spins.push_back(qs);
    }

    clust.clear();
    clust.reserve(root_to_cluster.size());
    for (auto& [root, cluster] : root_to_cluster){
        cluster.spins.shrink_to_fit();
        clust.push_back(std::move(cluster));
    }

    // link up the spins' "owning_cluster" pointers
    for (auto& qc : clust){
        for (auto& s : qc.spins){
            s->owning_cluster = &qc;
        }
    }

    // detect and store the boundary spins
    for (auto& qc : clust){
        std::unordered_set<Spin*> seen;
        for (auto& s : qc.spins){
            for (const auto nb : s->neighbours){
                if (nb->deleted) continue;
                if (nb->owning_cluster == nullptr && seen.insert(nb).second)
                    qc.classical_boundary_spins.push_back(nb);
                else if (nb->owning_cluster != nullptr
                        && nb->owning_cluster != &qc && seen.insert(nb).second)
                    qc.quantum_boundary_spins.push_back(nb);
            }
        }
    }
}


#pragma once
#include "quantum_cluster.hpp"
#include <unordered_map>
#include <unordered_set>



using CountCOO = std::tuple<size_t, size_t, int>;


// this class identifies all pairs of defects separated by length 6 or more

class TetraBondDFS {
    std::unordered_map<Tetra *, size_t> defect_tetras_map;
    const int max_depth;

    size_t curr_tetra_id;

    std::vector<std::vector<CountCOO>> connectivity_data;
    // first index: hop length-1 (0 = NN tetra pairs)
    // second index: entry index

    // visited: pointer set for the current DFS path, prevents revisiting nodes
    std::unordered_set<Tetra*> visited;

    void _dfs_traverse(Tetra *curr, int depth) {
        if (depth >= max_depth) return;

        for (const auto &next : curr->neighbours) {
            if (visited.count(next.tet)) continue;

            auto it = defect_tetras_map.find(next.tet);
            if (it != defect_tetras_map.end() && it->second < curr_tetra_id) {
                connectivity_data[depth].push_back({curr_tetra_id, it->second, 1});
            }

            visited.insert(next.tet);
            _dfs_traverse(next.tet, depth + 1);
            visited.erase(next.tet);
        }
    }

    public:
    TetraBondDFS(const std::unordered_set<Tetra *> &defect_tetras_,
            int max_depth_)
        : max_depth(max_depth_) {
            size_t idx = 0;
            for (auto t : defect_tetras_) {
                defect_tetras_map[t] = idx++;
            }

            connectivity_data.resize(max_depth);

            for (auto t : defect_tetras_) {
                curr_tetra_id = defect_tetras_map[t];
                visited.insert(t);
                _dfs_traverse(t, 0);
                visited.clear();
            }
        }

    const std::vector<std::vector<CountCOO>>& data() const {
        return connectivity_data;
    }
};


// pretty rough 
inline double Jfunc(double Jpm, int diamond_dist){
    switch (diamond_dist) {
    case 2:
        return Jpm;
    case 4:
        return -Jpm*Jpm;
    case 6:
        return Jpm*Jpm*Jpm;
    default:
        return 0;
    }
}


struct FrustrationStats {
    int    cycle_length;
    size_t n_total;
    size_t n_frustrated;
    double frust_fraction() const {
        return n_total ? double(n_frustrated) / double(n_total) : 0.0;
    }
};

struct Bond { size_t lo, hi; double J; };

// Finds all simple N-cycles in the defect-tetrahedra bond graph and classifies
// them by the Toulouse criterion: a cycle is frustrated if prod(J_ij) < 0.
//
// Bonds are built from dfs_traverser output at the requested hop depths
// (default {1,3,5} = 2-, 4-, 6-hop diamond paths). Multiple paths between
// the same pair at the same hop length are summed; bonds at different hop
// lengths are then summed into a single effective J per pair.
class CycleFrustration {
    // Keyed only on nodes that have at least one bond; isolated defects absent.
    std::unordered_map<size_t, std::vector<std::pair<size_t, double>>> adj_;

    const std::vector<std::pair<size_t,double>>& neighbours(size_t u) const {
        static const std::vector<std::pair<size_t,double>> empty;
        auto it = adj_.find(u);
        return it != adj_.end() ? it->second : empty;
    }

    // Single DFS pass covering all cycle lengths in [min_depth+1, max_depth+1].
    // At each eligible depth, tries to close back to start and records the result.
    void dfs(std::vector<size_t>& path, double J_prod,
             size_t start, int min_depth, int max_depth,
             std::vector<FrustrationStats>& stats) const
    {
        size_t curr  = path.back();
        int    depth = static_cast<int>(path.size()) - 1;

        auto& neigh = neighbours(curr);

        // Try to close the cycle at this depth if long enough
        if (depth >= min_depth) {
            for (auto& [nbr, J] : neigh) {
                if (nbr != start) continue;
                // Canonical orientation: count A-B-...-Z only when B < Z,
                // suppressing the reverse duplicate A-Z-...-B.
                if (path[1] < path.back()) {
                    double prod = J_prod * J;
                    auto& s = stats[depth - min_depth];
                    ++s.n_total;
                    if (prod < 0.0) ++s.n_frustrated;
                }
                break; // at most one edge to start after bond aggregation
            }
        }

        if (depth == max_depth) return;

        for (auto& [nbr, J] : neigh) {
            // start must remain the minimum-index node in the cycle
            if (nbr <= start) continue;
            bool in_path = std::any_of(path.begin(), path.end(),
                                       [nbr](size_t x){ return x == nbr; });
            if (in_path) continue;

            if (depth == max_depth - 1) {
                // only recurse if nbr can actually close the cycle
                bool closeable = false;
                for (auto& [n2, _] : neighbours(nbr))
                    if (n2 == start) { closeable = true; break; }
                if (!closeable) continue;
            }

            path.push_back(nbr);
            dfs(path, J_prod * J, start, min_depth, max_depth, stats);
            path.pop_back();
        }
    }

public:
    // hop_depths: indices into dfs_traverser::data() to include as bonds.
    // Default {1,3,5} selects the physically relevant even-hop paths
    // (2-, 4-, 6-hops on the diamond lattice).
    CycleFrustration(const std::vector<std::vector<CountCOO>>& bond_data,
                     double Jpm,
                     const std::vector<int>& hop_depths = {1, 3, 5})
    {
        // Aggregate: count paths per (lo, hi, hop_length)
        std::map<std::tuple<size_t,size_t,int>, int> path_counts;
        for (int d : hop_depths) {
            int hop = d + 1;
            for (auto& [a, b, _] : bond_data[d]) {
                size_t lo = std::min(a, b), hi = std::max(a, b);
                path_counts[{lo, hi, hop}]++;
            }
        }

        // Sum contributions from all hop lengths into one effective J per pair
        std::map<std::pair<size_t,size_t>, double> bond_J;
        for (auto& [key, n] : path_counts) {
            auto [lo, hi, hop] = key;
            bond_J[{lo, hi}] += n * Jfunc(Jpm, hop);
        }

        adj_.reserve(2 * bond_J.size()); // each bond touches at most 2 distinct nodes
        for (auto& [pair, J] : bond_J) {
            adj_[pair.first ].emplace_back(pair.second, J);
            adj_[pair.second].emplace_back(pair.first,  J);
        }
    }

    // Iterate only over the lo side of each bond (forward edges, no double-count).
    std::vector<Bond> bonds() const {
        std::vector<Bond> result;
        for (auto& [u, nbrs] : adj_)
            for (auto& [v, J] : nbrs)
                if (v > u)
                    result.push_back({u, v, J});
        return result;
    }

    // Returns one FrustrationStats per cycle length in [min_N, max_N].
    // A single DFS pass covers the whole range without redundant traversal.
    std::vector<FrustrationStats> compute(int min_N, int max_N) const {
        std::vector<FrustrationStats> stats;
        stats.reserve(max_N - min_N + 1);
        for (int n = min_N; n <= max_N; ++n)
            stats.push_back({n, 0, 0});

        std::vector<size_t> path;
        path.reserve(max_N);
        for (auto& [s, _] : adj_) {
            path.push_back(s);
            dfs(path, 1.0, s, min_N - 1, max_N - 1, stats);
            path.pop_back();
        }
        return stats;
    }
};


// Accumulates J_{ij} distribution statistics over multiple disorder realisations.
// Histogram bins are in units of Jpm; absolute bin edges = bin_edges() * Jpm.
struct JDistStats {
    static constexpr int    NBINS     = 200;
    static constexpr double HALFRANGE = 5.0; // histogram spans [-5,+5] * Jpm

    std::vector<int64_t> hist       = std::vector<int64_t>(NBINS, 0);
    std::vector<int64_t> degree_hist;           // degree_hist[k]: defect nodes with k bonds
    double  sum_J  = 0, sum_J2 = 0;
    int64_t n_bonds = 0, n_neg  = 0;

    void accumulate(const std::vector<Bond>& bonds, size_t n_defects, double Jpm) {
        std::vector<int> deg(n_defects, 0);
        for (auto& b : bonds) {
            double x   = b.J / Jpm;
            int    bin = static_cast<int>((x + HALFRANGE) / (2.0 * HALFRANGE) * NBINS);
            if (bin >= 0 && bin < NBINS) hist[bin]++;

            sum_J  += b.J;
            sum_J2 += b.J * b.J;
            ++n_bonds;
            if (b.J < 0.0) ++n_neg;

            deg[b.lo]++;
            deg[b.hi]++;
        }

        int max_deg = deg.empty() ? 0 : *std::max_element(deg.begin(), deg.end());
        if (static_cast<int>(degree_hist.size()) <= max_deg)
            degree_hist.resize(max_deg + 1, 0);
        for (int d : deg) degree_hist[d]++;
    }

    double mean()     const { return n_bonds ? sum_J  / n_bonds : 0.0; }
    double variance() const { return n_bonds ? sum_J2 / n_bonds - mean() * mean() : 0.0; }
    double neg_frac() const { return n_bonds ? double(n_neg) / n_bonds : 0.0; }

    // Bin edges in units of Jpm, length NBINS+1
    static std::vector<double> bin_edges() {
        std::vector<double> edges(NBINS + 1);
        for (int i = 0; i <= NBINS; ++i)
            edges[i] = -HALFRANGE + 2.0 * HALFRANGE * i / NBINS;
        return edges;
    }
};


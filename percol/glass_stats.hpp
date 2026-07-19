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
    size_t n_total      = 0;
    size_t n_frustrated = 0;
    double weight_total      = 0.0; // sum of sum_i|J_i| over all cycles
    double weight_frustrated = 0.0; // same, restricted to frustrated cycles
    double frust_fraction() const {
        return n_total ? double(n_frustrated) / double(n_total) : 0.0;
    }
    // Importance-sampled frustration: cycles weighted by sum_i|J_i|
    double weighted_frust_fraction() const {
        return weight_total > 0.0 ? weight_frustrated / weight_total : 0.0;
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
//
// Usage pattern for sweeping Jpm:
//   CycleFrustration cf(dfs_data.data());          // once per disorder realization
//   auto cycles = cf.enumerate_cycles(3, 4);       // expensive DFS, also once
//   for (double jpm : jpm_values) {
//       auto stats = cf.classify(cycles, jpm, 3, 4); // cheap per-Jpm
//       auto bds   = cf.bonds(jpm);                  // cheap per-Jpm
//   }
class CycleFrustration {
public:
    struct CycleBond { size_t lo, hi; };
    struct EnumeratedCycle { int length; std::vector<CycleBond> bonds; };

private:
    struct BondCounts { int n2 = 0, n4 = 0, n6 = 0; };

    // Path counts per canonical (lo < hi) bond pair
    std::map<std::pair<size_t,size_t>, BondCounts> bond_counts_;
    // Topology adjacency: both directions, used only for cycle enumeration
    std::unordered_map<size_t, std::vector<size_t>> adj_topo_;

    double J_from_counts(const BondCounts& bc, double Jpm) const {
        return bc.n2 * Jfunc(Jpm, 2) + bc.n4 * Jfunc(Jpm, 4) + bc.n6 * Jfunc(Jpm, 6);
    }

    // DFS for cycle enumeration — topology only, no J values.
    // Canonical orientation: only records cycle A-B-…-Z when B < Z.
    void dfs_enum(std::vector<size_t>& path, size_t start,
                  int min_depth, int max_depth,
                  std::vector<EnumeratedCycle>& cycles) const
    {
        size_t curr  = path.back();
        int    depth = static_cast<int>(path.size()) - 1;

        auto it = adj_topo_.find(curr);
        if (it == adj_topo_.end()) return;

        if (depth >= min_depth) {
            for (size_t nbr : it->second) {
                if (nbr != start) continue;
                if (path[1] < path.back()) {
                    EnumeratedCycle cyc;
                    cyc.length = depth + 1;
                    cyc.bonds.reserve(depth + 1);
                    for (int k = 0; k < depth; ++k) {
                        size_t a = path[k], b = path[k+1];
                        cyc.bonds.push_back({std::min(a,b), std::max(a,b)});
                    }
                    cyc.bonds.push_back({std::min(path.back(), start),
                                         std::max(path.back(), start)});
                    cycles.push_back(std::move(cyc));
                }
                break;
            }
        }

        if (depth == max_depth) return;

        for (size_t nbr : it->second) {
            if (nbr <= start) continue;

            if (depth == max_depth - 1) {
                bool closeable = false;
                auto it2 = adj_topo_.find(nbr);
                if (it2 != adj_topo_.end())
                    for (size_t n2 : it2->second)
                        if (n2 == start) { closeable = true; break; }
                if (!closeable) continue;
            }

            path.push_back(nbr);
            dfs_enum(path, start, min_depth, max_depth, cycles);
            path.pop_back();
        }
    }

public:
    // hop_depths: indices into dfs_traverser::data() to include as bonds.
    // Default {1,3,5} selects the physically relevant even-hop paths
    // (2-, 4-, 6-hops on the diamond lattice).
    CycleFrustration(const std::vector<std::vector<CountCOO>>& bond_data,
                     const std::vector<int>& hop_depths = {1, 3, 5})
    {
        std::map<std::tuple<size_t,size_t,int>, int> path_counts;
        for (int d : hop_depths) {
            int hop = d + 1;
            for (auto& [a, b, _] : bond_data[d]) {
                size_t lo = std::min(a, b), hi = std::max(a, b);
                path_counts[{lo, hi, hop}]++;
            }
        }

        for (auto& [key, n] : path_counts) {
            auto [lo, hi, hop] = key;
            auto& bc = bond_counts_[{lo, hi}];
            if      (hop == 2) bc.n2 += n;
            else if (hop == 4) bc.n4 += n;
            else if (hop == 6) bc.n6 += n;
        }

        for (auto& [pair, bc] : bond_counts_) {
            adj_topo_[pair.first ].push_back(pair.second);
            adj_topo_[pair.second].push_back(pair.first );
        }
    }

    // Enumerate all simple cycles of length [min_N, max_N]. Expensive: O(N * paths).
    // Call once per disorder realization and reuse for multiple Jpm values.
    std::vector<EnumeratedCycle> enumerate_cycles(int min_N, int max_N) const {
        std::vector<EnumeratedCycle> cycles;
        std::vector<size_t> path;
        path.reserve(max_N);
        for (auto& [s, _] : adj_topo_) {
            path.push_back(s);
            dfs_enum(path, s, min_N - 1, max_N - 1, cycles);
            path.pop_back();
        }
        return cycles;
    }

    // Evaluate frustration for a specific Jpm from pre-enumerated cycles. Cheap.
    std::vector<FrustrationStats> classify(const std::vector<EnumeratedCycle>& cycles,
                                           double Jpm, int min_N, int max_N) const
    {
        std::vector<FrustrationStats> stats;
        stats.reserve(max_N - min_N + 1);
        for (int n = min_N; n <= max_N; ++n) stats.push_back({n, 0, 0});

        for (auto& cyc : cycles) {
            int idx = cyc.length - min_N;
            if (idx < 0 || idx >= static_cast<int>(stats.size())) continue;
            auto& s = stats[idx];
            ++s.n_total;
            double J_prod = 1.0;
            double weight = std::numeric_limits<double>::max();
            for (auto& b : cyc.bonds) {
                auto it = bond_counts_.find({b.lo, b.hi});
                if (it != bond_counts_.end()) {
                    double J = J_from_counts(it->second, Jpm);
                    J_prod *= -J;
                    weight = std::min(weight, std::abs(J));
                }
            }
            s.weight_total += weight;
            if (J_prod < 0.0) {
                ++s.n_frustrated;
                s.weight_frustrated += weight;
            }
        }
        return stats;
    }

    // Build bond list with J values for a specific Jpm. Cheap.
    std::vector<Bond> bonds(double Jpm) const {
        std::vector<Bond> result;
        result.reserve(bond_counts_.size());
        for (auto& [pair, bc] : bond_counts_)
            result.push_back({pair.first, pair.second, J_from_counts(bc, Jpm)});
        return result;
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


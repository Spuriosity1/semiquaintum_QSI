#pragma once

#include "lattice_lib/unitcell_types.hpp"
#include <cmath>
#include <vector>

namespace pyrochlore {
// nn_vectors[sl] -> neighbour vectors by spin SL
const std::vector<std::vector<ipos_t>> nn_vectors = {
    {{0,2,2},{2,0,2},{2,2,0},{0,-2,-2},{-2,0,-2},{-2,-2,0}},
    {{2, -2, 0}, {2, 0, -2}, {0, -2, -2}, {-2, 2, 0}, {-2, 0, 2}, {0, 2, 
2}},
    {{0, 2, -2}, {-2, 0, -2}, {-2, 2, 0}, {0, -2, 2}, {2, 0, 2}, {2, -2, 
0}},
    {{-2, -2, 0}, {-2, 0, 2}, {0, -2, 2}, {2, 2, 0}, {2, 0, -2}, {0, 
2, -2}}
};

const std::vector<ipos_t> pyro {
    {-1,-1,-1},
    {-1,1,1},
    {1,-1,1},
    {1,1,-1}
}; // vectors from cell origin to the pyro position


const std::vector<ipos_t> fcc {
    {0,0,0},
    {0,4,4},
    {4,0,4},
    {4,4,0}
}; // vectors from cell origin to the pyro position


const std::vector<ipos_t> plaq_boundaries[4] = {
    {
        {0, -2, 2},    {2, -2, 0},
        {2, 0, -2},    {0, 2, -2}, 
        {-2, 2, 0},    {-2, 0, 2}
    },
    {
        { 0, 2,-2},    { 2, 2, 0},
        { 2, 0, 2},    { 0,-2, 2},
        {-2,-2, 0},    {-2, 0,-2}
    },
    {
        { 0,-2,-2},	{-2,-2, 0},
        {-2, 0, 2},	{ 0, 2, 2},
        { 2, 2, 0},	{ 2, 0,-2}
    },
    {
        { 0, 2, 2},	{-2, 2, 0},
        {-2, 0,-2},	{ 0,-2,-2},
        { 2,-2, 0},	{ 2, 0, 2}
    }
};


// Returns one normalized local quantization axis per lil2 Spin sublattice.
// The pyrochlore lattice has 4 distinct [111]-type local axes; sublattice sl
// in the lil2 supercell maps to pyro[sl % 4] (the order in which spins are
// added in initialise_lattice()).
// Pass n_sublattices = std::get<SlPos<Spin>>(sc.sl_positions).size().
inline std::vector<vector3::vec3d> pyrochlore_local_axes() {
    const double inv3 = 1.0 / std::sqrt(3.0);
    std::vector<vector3::vec3d> axes;
    axes.reserve(4);
    for (int sl = 0; sl < 4; sl++) {
        const auto& p = pyro[sl];
        axes.emplace_back(p[0] * inv3, p[1] * inv3, p[2] * inv3);
    }
    return axes;
}

}; // end namespace

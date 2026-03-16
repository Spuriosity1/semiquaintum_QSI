#pragma once

#include "lattice_lib/unitcell_types.hpp"
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


}; // end namespace

#pragma once 

#include <vector>
#include "unitcell_types.hpp"

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
}; // vectos from cell origin to the pyro position



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

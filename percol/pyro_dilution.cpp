
#include <random>
#include "unionfind.hpp"
#include <argparse/argparse.hpp>


// finds the label for the cluster of element, 'elem'
template <UFElement T> 
T* find(T* elem, ipos_t& path_dx){
    path_dx={0,0,0};
    T* curr = elem;
    while (curr->parent != curr){
        path_dx += curr->dx;
        curr = curr->parent;
    }
    // curr is now the root node
    return curr;
}

template <UFElement T> 
T* find_and_compress(T* elem, ipos_t& path_dx){
    if (elem->parent == elem) {
        path_dx = {0,0,0};
        return elem;
    }

  // Recursively find root and get displacement from parent to root
    ipos_t parent_dx;
    T* root = find_and_compress(elem->parent, parent_dx);
    
    // Update displacement and compress path
    path_dx = elem->dx + parent_dx;
    elem->dx = path_dx;
    elem->parent = root;

    return root;
}

// Joins e1 to e2. 
// returns true if the resulting join created a winding path
template <UFElement T>
bool join_nodes(T* e1, T* e2, const LatticeIndexing& lat){
    ipos_t dx1{0,0,0};
    ipos_t dx2{0,0,0};
    T* root1 = find_and_compress(e1, dx1);
    T* root2 = find_and_compress(e2, dx2);

    auto Delta_x =  e2->ipos - e1->ipos;
    lat.wrap_super_delta(Delta_x);

    bool percolates = false;

    if (root1 == root2){
        // check for winding!
        auto loop_dx = Delta_x - (dx2 - dx1);
//        std::cout<<"loop_dx = "<<loop_dx<<"\t Dx="<<Delta_x<<"\tdx1-dx2="<<dx1-dx2<<"\n";
        if (loop_dx != ipos_t{0,0,0}){
            percolates = true;
        }
    } else {
        // merge the two
        root2->parent = root1;
        root2->dx = Delta_x - (dx2 - dx1);
    }
    
    return percolates;
}



// modifies the "root" parts of the elements; union-find with path compression
template<UFElement T> 
bool initialise_tree(std::vector<T>& elements, const LatticeIndexing& lat
        ){
    // reset all spins
    for (auto& el : elements){
        el.parent = std::addressof(el);
        el.dx = {0,0,0};
    }

    bool percolating = false;

    for (auto& el : elements){
        if (el.deleted) continue;

        // check neighbours
        for (auto s : el.neighbours){
            if (s->deleted) continue;
            percolating |= join_nodes(s, &el, lat);
        }
    }

    return percolating;
}


using namespace std;

int main (int argc, char *argv[]) {

    auto ap = argparse::ArgumentParser("pyro_percol");
    ap.add_argument("L")
        .help("Linear dimension of the system")
        .scan<'i', int>();
    ap.add_argument("p")
        .help("Dilution probability")
        .scan<'g', float>();

    ap.add_argument("--seed", "-s")
        .help("RNG seed")
        .scan<'i', size_t>();
    ap.add_argument("--nsweep", "-w")
        .help("Iterations in the RNG sweep")
        .scan<'i', size_t>();

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t seed = ap.get<size_t>("--seed");
    size_t nsweep = ap.get<size_t>("--nsweep");

    QClattice sc = initialise_lattice(L);
//    cerr<<"Latvecs: "<<sc.lattice.get_lattice_vectors()<<"\n";

    std::vector<int> links_percolate; // stores 1=percolates, 
                                  // 0=does not percolate

    // Delete about p*100% of the spins
    // (Bernoulli sample)
    std::mt19937 rng(seed);


    for (size_t i=0; i<nsweep; i++){
        spin_sweep(sc, p, rng);
        bool links_per = initialise_tree(sc.get_objects<Spin>(), sc.lattice);
        links_percolate.push_back(links_per);
    }

    size_t n_pecol = std::accumulate(links_percolate.begin(), links_percolate.end(), static_cast<size_t>(0));

    size_t n_percol2 =0;
    for ( auto p : links_percolate ){
        n_percol2 += p*p;
    }
    double mu = n_pecol*1.0/nsweep;
    double var = n_percol2 *1.0 / nsweep - mu*mu;

    std::cout<<"Percolation probability: "<<mu<<" +- "<<sqrt(var)<<std::endl;


    return 0;
}


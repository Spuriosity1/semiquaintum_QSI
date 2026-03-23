#pragma once

#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include "quantum_cluster.hpp"
#include <cstddef>
#include <filesystem>
#include <vector>


// Measures the spinon texture at all MC temperatures
class Q_manager {
    std::vector<double> T_list;
    std::vector<std::array<double, 4>> spinon2_expectation;
    std::vector<std::array<size_t, 4>> count;
    // spinon2_expectation[n] -> <Q^2> on tetras with 4-n sites
    // spinon2_expectation[0] -> complete tetras
    // spinon2_expectation[1] -> triangles
    // spinon2_expectation[2] -> lines
    // spinon2_expectation[3] -> dangling spins
    

    public:
    Q_manager(size_t n_reserve=0) {
        T_list.reserve(n_reserve);
        spinon2_expectation.reserve(n_reserve);   
        
    }

    void new_T(double T){
        T_list.push_back(T);
        spinon2_expectation.push_back({0,0,0,0});
        count.push_back({0,0,0,0});
    }

    // TODO: should be const but Supercell has no const accessor
    void sample(QClattice& sc ){
        auto& acc = spinon2_expectation.back();
        auto& count_acc = count.back();
        for (const auto& t : sc.get_objects<Tetra>()){
            double Q=0;
            for (auto s : t.member_spins){
                if (s->deleted) continue;
                if (s->is_quantum()) {
                    auto qc = static_cast<QClusterMF*>(s->owning_cluster);
                    Q += qc->expect_Sz(s);
                } else {
                    Q += s->ising_val;
                }
            }
            Q/=2;
            acc[4-t.neighbours.size()] += Q*Q;
            count_acc[4-t.neighbours.size()]++;
        }
    }


    // Mean Q^2 per defect type for the most recent temperature.
    // Returns {complete, triangle, line, dangling}; entry is NaN if no tetras of that type.
    std::array<double, 4> curr_Q2() const {
        const auto& s = spinon2_expectation.back();
        const auto& c = count.back();
        std::array<double, 4> out;
        for (int i = 0; i < 4; i++)
            out[i] = c[i] ? s[i] / c[i] : std::numeric_limits<double>::quiet_NaN();
        return out;
    }

    void save(const std::filesystem::path& file_path);
    void write_group(hid_t file_id, const char* group_name="/charge");
};


inline void Q_manager::save(const std::filesystem::path& file_path) {
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    write_group(file_id);
    H5Fclose(file_id);
}

inline void Q_manager::write_group(hid_t file_id, const char* group_name) {
    hid_t group = H5Gcreate2(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group < 0)
        throw std::runtime_error("Failed to create group");

    const hsize_t n = T_list.size();

    auto write_1d = [&](const char* name, hid_t type, const void* data) {
        hsize_t dims[1] = {n};
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(group, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
        H5Sclose(space);
    };

    write_1d("T_list", H5T_NATIVE_DOUBLE, T_list.data());

    // spinon2_expectation: stored as (n_temperatures, 4).
    // std::vector<std::array<double,4>> is contiguous, so .data() is a flat double[n][4].
    // Rows: temperatures; columns: defect type (0=complete, 1=triangle, 2=line, 3=dangling).
    {
        hsize_t dims[2] = {n, 4};
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(group, "spinon2_expectation", H5T_NATIVE_DOUBLE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 spinon2_expectation.data());
        H5Dclose(ds);
        H5Sclose(space);
    }

    H5Gclose(group);
}

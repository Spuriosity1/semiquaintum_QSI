#pragma once

#include "H5Gpublic.h"
#include "H5Ipublic.h"
#include "H5Ppublic.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <numeric>
#include <vector>


class energy_manager {
    std::vector<double> E;
    std::vector<double> E2;
    std::vector<double> T_list;
    std::vector<size_t> n_samples;
    size_t curr_idx=0;

    public:
    energy_manager(size_t n_temperatures_reserve=0) {
        E.reserve(n_temperatures_reserve);
        E2.reserve(n_temperatures_reserve);
        T_list.reserve(n_temperatures_reserve);
        n_samples.reserve(n_temperatures_reserve);
    }

    void new_T(double T){
        curr_idx=T_list.size();

        E.push_back(0);
        E2.push_back(0);
        n_samples.push_back(0);
        T_list.push_back(T);
    }

    void set_T(double T, double tol=1e-8){
        for (size_t i = 0; i < T_list.size(); i++){
            if (std::abs(T_list[i] - T) < tol){
                curr_idx = i;
                return;
            }
        }
        // Not found — create new entry.
        curr_idx = T_list.size();
        E.push_back(0);
        E2.push_back(0);
        n_samples.push_back(0);
        T_list.push_back(T);
    }

    double curr_T(){
        return T_list[curr_idx];
    }


    double curr_E() const {
        return E[curr_idx] / n_samples[curr_idx];
    }
        
    void sample(double _e){
        assert(!T_list.empty());

        E[curr_idx] += _e;
        E2[curr_idx] += (_e*_e);
        n_samples[curr_idx]++;
    }

    void save(const std::filesystem::path& file_path);
    void write_group(hid_t file_id, const char* group_name="/energy");

    // Truncate (or create) the HDF5 file at file_path so that subsequent
    // write_group() calls append into a clean file.  Call once before the
    // first write_group() when the same file is reused across runs or replicas.
    static void init_file(const std::filesystem::path& file_path) {
        hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC,
                                   H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0)
            throw std::runtime_error("Failed to initialise HDF5 file: " + file_path.string());
        H5Fclose(file_id);
    }
};



inline void energy_manager::save(const std::filesystem::path& file_path){

    // Create HDF5 file
    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }
    write_group(file_id);
   
    // Close groups and file
    H5Fclose(file_id);
}

inline void energy_manager::write_group(hid_t file_id, const char* group_name){
    // Sort all arrays by temperature (ascending) before writing.
    const size_t n = T_list.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return T_list[a] < T_list[b]; });

    std::vector<double>  sT(n), sE(n), sE2(n);
    std::vector<size_t>  sN(n);
    for (size_t i = 0; i < n; i++){
        sT[i]  = T_list[idx[i]];
        sE[i]  = E[idx[i]];
        sE2[i] = E2[idx[i]];
        sN[i]  = n_samples[idx[i]];
    }

    // Open existing group or create a new one.
    hid_t data_group;
    if (H5Lexists(file_id, group_name, H5P_DEFAULT) > 0) {
        data_group = H5Gopen2(file_id, group_name, H5P_DEFAULT);
    } else {
        data_group = H5Gcreate2(file_id, group_name,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (data_group < 0)
        throw std::runtime_error(std::string("Failed to open/create group ") + group_name);

    const hsize_t n_new = n;

    // Append n_new entries to dataset `name`.  If the dataset does not exist
    // yet it is created with chunked/unlimited storage so future appends work.
    auto append_dataset = [&](const char* name, hid_t type, const void* data) {
        if (H5Lexists(data_group, name, H5P_DEFAULT) > 0) {
            // Dataset exists — extend it and write new data at the end.
            hid_t ds = H5Dopen2(data_group, name, H5P_DEFAULT);
            if (ds < 0) throw std::runtime_error(std::string("Failed to open dataset ") + name);

            hid_t fspace = H5Dget_space(ds);
            hsize_t cur[1];
            H5Sget_simple_extent_dims(fspace, cur, nullptr);
            H5Sclose(fspace);

            const hsize_t new_size[1] = { cur[0] + n_new };
            H5Dset_extent(ds, new_size);

            fspace = H5Dget_space(ds);
            const hsize_t offset[1] = { cur[0] };
            const hsize_t count[1]  = { n_new };
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            hid_t mspace = H5Screate_simple(1, count, nullptr);
            herr_t st = H5Dwrite(ds, type, mspace, fspace, H5P_DEFAULT, data);
            H5Sclose(mspace);
            H5Sclose(fspace);
            H5Dclose(ds);
            if (st < 0) throw std::runtime_error(std::string("Failed to append dataset ") + name);
        } else {
            // Dataset does not exist — create with chunked/unlimited dimensions.
            const hsize_t maxdims[1] = { H5S_UNLIMITED };
            const hsize_t chunk[1]   = { n_new > 0 ? n_new : 1 };
            hid_t space = H5Screate_simple(1, &n_new, maxdims);
            hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, chunk);
            hid_t ds = H5Dcreate2(data_group, name, type, space,
                                   H5P_DEFAULT, dcpl, H5P_DEFAULT);
            if (ds < 0) {
                H5Pclose(dcpl); H5Sclose(space); H5Gclose(data_group);
                throw std::runtime_error(std::string("Failed to create dataset ") + name);
            }
            herr_t st = H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Pclose(dcpl);
            H5Sclose(space);
            H5Dclose(ds);
            if (st < 0) throw std::runtime_error(std::string("Failed to write dataset ") + name);
        }
    };

    append_dataset("E",         H5T_NATIVE_DOUBLE,  sE.data());
    append_dataset("E2",        H5T_NATIVE_DOUBLE,  sE2.data());
    append_dataset("T_list",    H5T_NATIVE_DOUBLE,  sT.data());
    append_dataset("n_samples", H5T_NATIVE_ULLONG,  sN.data());

    H5Gclose(data_group);
}


#pragma once

#include "H5Ipublic.h"
#include <cmath>
#include <cstddef>
#include <vector>

// Common base for per-temperature observable managers.
//
// Provides T_list / n_samples bookkeeping and the new_T / set_T interface.
// Derived classes override on_new_temp() to push their own per-temperature
// storage whenever a new temperature slot is created.
class abstract_manager {
protected:
    std::vector<double> T_list;
    std::vector<size_t> n_samples;
    size_t curr_idx = 0;

    // Called immediately after a new temperature entry is appended.
    // Override to push per-temperature storage (e.g. E.push_back(0)).
    virtual void on_new_temp() {}

public:
    virtual ~abstract_manager() = default;

    // Append a new temperature slot and make it current.
    void new_T(double T) {
        curr_idx = T_list.size();
        T_list.push_back(T);
        n_samples.push_back(0);
        on_new_temp();
    }

    // Switch to an existing temperature (within tol), or create a new slot.
    void set_T(double T, double tol = 1e-8) {
        for (size_t i = 0; i < T_list.size(); i++) {
            if (std::abs(T_list[i] - T) < tol) {
                curr_idx = i;
                return;
            }
        }
        curr_idx = T_list.size();
        T_list.push_back(T);
        n_samples.push_back(0);
        on_new_temp();
    }

    double curr_T() const { return T_list[curr_idx]; }

    virtual void write_group(hid_t file_id, const char* group_name) = 0;
};

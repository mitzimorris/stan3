#ifndef STAN3_RUN_OTHERS_HPP
#define STAN3_RUN_OTHERS_HPP

// #include <stan3/arguments.hpp>
// #include <stan/run/config_model_inits.hpp>
// #include <stan/run/algorithm_type.hpp>
// #include <stan/run/config_algorithm.hpp>

#include <CLI11/CLI11.hpp>
#include <string>
#include <map>
#include <memory>


namespace stan3 {

// Function to run MLE algorithm
  inline int run_mle() { 
    std::cout << "Running MLE algorithm (not implemented)" << std::endl;
    return 0;
}

// Function to run Pathfinder algorithm
inline int run_pathfinder() {
    std::cout << "Running Pathfinder algorithm (not implemented)" << std::endl;
    return 0;
}

// Function to run ADVI algorithm
inline int run_advi() {
    std::cout << "Running ADVI algorithm (not implemented)" << std::endl;
    return 0;
}

// Function to run Generate Quantities
inline int run_gq() {
    std::cout << "Running Generate Quantities (not implemented)" << std::endl;
    return 0;
}

}  // namespace stan3

#endif  //

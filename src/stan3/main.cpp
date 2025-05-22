#include <iostream>
#include <string>
#include <CLI11/CLI11.hpp>
#include <stan3/algorithm_type.hpp>
#include <stan3/hmc_nuts_arguments.hpp>
#include <stan3/metric_type.hpp>
#include <stan3/run_hmc_nuts.hpp>
#include <stan3/run_others.hpp>

int main(int argc, char** argv) {
    // Create CLI app
    CLI::App app{"Stan3 - Command line interface for Stan"};
    
    // Arguments struct to store parsed values
    stan3::hmc_nuts_args args;
    
    // Setup CLI options
    stan3::setup_cli(app, args);
    
    // Parse command line
    CLI11_PARSE(app, argc, argv);

    std::string error_message;
    if (!stan3::validate_arguments(args, error_message)) {
        std::cerr << error_message << std::endl;
        return 1;
    }
    stan3::finalize_arguments(args);
    std::cout << "config" << std::endl << app.config_to_str() << std::endl;

    // Dispatch to the appropriate algorithm
    switch (args.algorithm) {
        case stan3::algorithm_t::STAN2_HMC:
            return stan3::run_hmc(args);
        
        case stan3::algorithm_t::MLE:
            return stan3::run_mle();
        
        case stan3::algorithm_t::PATHFINDER:
            return stan3::run_pathfinder();
        
        case stan3::algorithm_t::ADVI:
            return stan3::run_advi();
        
        case stan3::algorithm_t::STANDALONE_GQ:
            return stan3::run_gq();
        
        default:
            std::cerr << "Error: Unknown algorithm selected" << std::endl;
            return 1;
    }
}

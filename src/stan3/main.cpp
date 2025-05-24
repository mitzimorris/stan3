#include <iostream>
#include <string>
#include <CLI11/CLI11.hpp>
#include <stan3/algorithm_type.hpp>
#include <stan3/arguments.hpp>
#include <stan3/load_model.hpp>
#include <stan3/metric_type.hpp>
#include <stan3/run_hmc_nuts.hpp>
#include <stan3/run_others.hpp>

int main(int argc, char** argv) {
    // Create CLI app
    CLI::App app{"Stan3 - Command line interface for Stan"};
    
    // Arguments structs
    stan3::stan3_args base_args;
    stan3::hmc_nuts_args hmc_args;
    
    // Setup CLI options and subcommands
    CLI::App* selected_subcommand = stan3::setup_cli(app, base_args, hmc_args);
    
    // Parse command line
    CLI11_PARSE(app, argc, argv);

    // Determine which algorithm was selected and validate accordingly
    std::string error_message;
    bool validation_passed = true;
    
    if (app.got_subcommand("hmc")) {
        validation_passed = stan3::validate_hmc_arguments(hmc_args, error_message);
        stan3::finalize_arguments(hmc_args);
    }
    // Add validation for other algorithms here as they're implemented
    
    if (!validation_passed) {
        std::cerr << error_message << std::endl;
        return 1;
    }
    
    std::cout << "config" << std::endl << app.config_to_str() << std::endl;

    // Load model (using base_args for common options like data_file)
    stan::model::model_base& model = stan3::load_model(
        app.got_subcommand("hmc") ? 
        static_cast<const stan3::stan3_args&>(hmc_args) : base_args);

    // Dispatch to the appropriate algorithm
    if (app.got_subcommand("hmc")) {
        return stan3::run_hmc(hmc_args, model);
    } else {
        // Handle other algorithms when they're implemented
        std::cerr << "Error: No algorithm subcommand selected" << std::endl;
        return 1;
    }
}

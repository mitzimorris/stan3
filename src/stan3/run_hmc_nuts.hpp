#ifndef STAN3_RUN_HMC_HPP
#define STAN3_RUN_HMC_HPP

#include <stan3/algorithm_type.hpp>
#include <stan3/hmc_nuts_arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan3/load_samplers.hpp>
#include <stan3/run_samplers.hpp>
#include <stan3/metric_type.hpp>
#include <stan3/read_json_data.hpp>

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>

#include <boost/random/mixmax.hpp>

#include <CLI11/CLI11.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>

using rng_t = boost::random::mixmax;

namespace stan3 {

/* Function to run HMC algorithm */
template <bool Jacobian = true, class Model>
int run_hmc(const hmc_nuts_args& args, Model& model) {
  std::stringstream err_msg;
  try {
    stan::callbacks::interrupt interrupt;
    stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                         std::cerr, std::cerr);
    std::string model_name = model.model_name();

    // Configure outputs
    std::vector<hmc_nuts_writers> writers;
    if (args.num_chains == 1) {
      auto timestamp = generate_timestamp();
      writers.push_back(create_hmc_nuts_single_chain_writers(
        args, model_name, timestamp, 1));
    } else {
      writers = create_hmc_nuts_multi_chain_writers(args, model_name);
    }

    std::vector<std::string> uparam_names;
    model.unconstrained_param_names(uparam_names, false, false);

    if (uparam_names.empty()) {
      // Handle fixed parameter models
      logger.info("Model has no parameters. Running fixed parameter sampler.");
      
      // TODO: Implement fixed parameter sampling if needed
      // For now, just return success
      std::cout << "Fixed parameter model detected - no sampling required." << std::endl;
      return 0;
    } else {
      // assemble initial param values, initial inverse metric
      std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts;
      init_contexts.reserve(args.num_chains);
      for (size_t i = 0; i < args.num_chains; ++i) {
        std::string init_file = get_init_file_for_chain(args, i);
        try {
          auto init_context = stan3::read_json_data(init_file);
          init_contexts.push_back(init_context);
        } catch (const std::exception &e) {
          err_msg << "Error reading initial parameter values file for chain " 
                  << (i + 1) << ": " << e.what() << std::endl;
          throw std::invalid_argument(err_msg.str());
        }
      }
      std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts;
      metric_contexts.reserve(args.num_chains);
      for (size_t i = 0; i < args.num_chains; ++i) {
        std::string metric_file = get_metric_file_for_chain(args, i);
        try {
          auto metric_context = stan3::read_json_data(metric_file);
          metric_contexts.push_back(metric_context);
        } catch (const std::exception &e) {
          err_msg << "Error reading precomputed inverse metric file for chain " 
                  << (i + 1) << ": " << e.what() << std::endl;
          throw std::invalid_argument(err_msg.str());
        }
      }
      try {
        run_samplers(model, args, init_contexts, metric_contexts, 
                    writers, interrupt, logger);
      } catch (const std::exception& e) {
        err_msg << "Error running samplers: " << e.what() << std::endl;
        throw std::runtime_error(err_msg.str());
      }
    }

    std::cout << "Sampling completed successfully!" << std::endl;
    std::cout << "  Output dir: " << args.output_dir << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace stan3
#endif  // STAN3_RUN_HMC_HPP

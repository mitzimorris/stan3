#ifndef STAN3_RUN_HMC_HPP
#define STAN3_RUN_HMC_HPP

#include <stan3/algorithm_type.hpp>
#include <stan3/hmc_nuts_arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan3/metric_type.hpp>
#include <stan3/read_json_data.hpp>
#include <stan3/load_model.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <boost/random/mixmax.hpp>

#include <CLI11/CLI11.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>

using rng_t = boost::random::mixmax;

namespace stan3 {

/* Function to run HMC algorithm
 *
 * @tparam Jacobian indicates whether to include the Jacobian term when
 *   evaluating the log density function
 * @tparam RNG the type of the random number generator
 */
template <bool Jacobian = true>
int run_hmc(const hmc_nuts_args& args) {
  std::stringstream err_msg;
  try {
    stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                         std::cerr, std::cerr);
    // Instantiate rngs
    std::vector<rng_t> rngs;
    for (unsigned int i = 0; i < args.num_chains; ++i) {
      rngs.push_back(stan::services::util::create_rng(args.random_seed, i));
    }

    // Read and validate data
    std::shared_ptr<const stan::io::var_context> data_context;
    try {
      data_context = stan3::read_json_data(args.data_file);
    } catch (const std::exception &e) {
      err_msg << "Error reading input data, "
              << e.what() << std::endl;
      throw std::invalid_argument(err_msg.str());
    }

    // Load model
    stan::io::var_context* raw_context = const_cast<stan::io::var_context*>(data_context.get());
    auto model = load_model(*raw_context, args.random_seed);
    std::string model_name = model->model_name();

    // Configure outputs
    std::vector<hmc_nuts_writers> writers;
    if (args.num_chains == 1) {
      auto timestamp = generate_timestamp();
      writers.push_back(create_hmc_nuts_single_chain_writers(
        args, model_name, timestamp, 1));
    } else {
      writers = create_hmc_nuts_multi_chain_writers(args, model_name);
    }


    // Read init contexts per chain
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
      // throws exception on failure
    }

    stan::model::model_base* raw_model = const_cast<stan::model::model_base*>(model.get());
    stan::callbacks::writer dummy_writer;
    std::vector<std::vector<double>> init_params;
    for (size_t i = 0; i < args.num_chains; ++i) {
      stan::io::var_context* init_context =
	const_cast<stan::io::var_context*>(init_contexts[i].get());


      auto inits = stan::services::util::initialize(*raw_model, *init_context, rngs[i],
						    args.init_radius, false, logger,
						    dummy_writer);
      init_params.push_back(inits);
    }


    // Read metric contexts per chain
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

        
    // Run the algorithm with these configurations
    std::cout << "Running HMC with:" << std::endl;
    std::cout << "  Chains: " << args.num_chains << std::endl;
    std::cout << "  Output dir: " << args.output_dir << std::endl;
    std::cout << "  Adapt delta: " << args.delta << std::endl;
    std::cout << "  Init files: " << args.init_files.size() << std::endl;
    std::cout << "  Metric files: " << args.metric_files.size() << std::endl;
        
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace stan3

#endif  // STAN3_RUN_HMC_HPP

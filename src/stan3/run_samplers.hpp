#ifndef STAN3_RUN_SAMPLERS_HPP
#define STAN3_RUN_SAMPLERS_HPP

#include <stan3/arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan3/load_samplers.hpp>
#include <stan3/metric_type.hpp>

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace stan3 {

template <typename Model>
class sampler_runner {
public:
  sampler_runner(Model& model, const hmc_nuts_args& args,
                 const std::vector<hmc_nuts_writers>& writers,
                 stan::callbacks::interrupt& interrupt,
                 stan::callbacks::logger& logger)
    : model_(model), args_(args), writers_(writers), 
      interrupt_(interrupt), logger_(logger) {}

  template <typename ConfigType>
  void operator()(ConfigType& config) {
    if (args_.base.num_chains == 1) {
      run_single_chain(config, 0);
    } else {
      run_multiple_chains_sequential(config);
    }
  }

private:
  template <typename ConfigType>
  void run_single_chain(ConfigType& config, size_t chain_idx) {
    auto& sampler = config.samplers[chain_idx];
    auto& init_params = config.init_params[chain_idx];
    auto& rng = config.rngs[chain_idx];
    
    // Get writers - handle nullable cases
    stan::callbacks::writer dummy_writer;
    stan::callbacks::structured_writer dummy_structured_writer;
    
    stan::callbacks::writer* diagnostic_writer = 
      writers_[chain_idx].diagnostics_writer ? 
      writers_[chain_idx].diagnostics_writer.get() : &dummy_writer;
    
    stan::callbacks::structured_writer* metric_writer = 
      writers_[chain_idx].metric_writer ? 
      writers_[chain_idx].metric_writer.get() : &dummy_structured_writer;
    
    stan::services::util::run_adaptive_sampler(
      sampler, model_, init_params, args_.num_warmup, args_.num_samples,
      args_.thin, args_.refresh, args_.save_warmup, rng, interrupt_, logger_,
      *writers_[chain_idx].sample_writer, *diagnostic_writer, *metric_writer, 
      chain_idx + 1, args_.base.num_chains);
  }
  
  template <typename ConfigType>
  void run_multiple_chains_sequential(ConfigType& config) {
    // Simple sequential loop - run each chain one after another
    for (size_t i = 0; i < args_.base.num_chains; ++i) {
      std::cout << "Starting chain " << (i + 1) << " of " << args_.base.num_chains << std::endl;
      
      try {
        run_single_chain(config, i);
        std::cout << "Completed chain " << (i + 1) << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "Chain " << (i + 1) << " failed: " << e.what() << std::endl;
        throw;
      }
    }
    std::cout << "All " << args_.base.num_chains << " chains completed successfully." << std::endl;
  }

  Model& model_;
  const hmc_nuts_args& args_;
  const std::vector<hmc_nuts_writers>& writers_;
  stan::callbacks::interrupt& interrupt_;
  stan::callbacks::logger& logger_;
};

/* Convenience function to create and run samplers */
  // Create init writers from the writers struct
  // Create samplers
  // Run samplers using visitor pattern
template <typename Model>
void run_samplers(Model& model,
                  const hmc_nuts_args& args,
                  const std::vector<std::shared_ptr<const stan::io::var_context>>& init_contexts,
                  const std::vector<std::shared_ptr<const stan::io::var_context>>& metric_contexts,
                  const std::vector<hmc_nuts_writers>& writers,
                  stan::callbacks::interrupt& interrupt,
                  stan::callbacks::logger& logger) {
  
  std::vector<stan::callbacks::writer*> init_writers;
  init_writers.reserve(args.base.num_chains);
  
  for (size_t i = 0; i < args.base.num_chains; ++i) {
    init_writers.push_back(
      writers[i].start_params_writer ? writers[i].start_params_writer.get() : nullptr);
  }
  auto sampler_configs = create_samplers(model, args, init_contexts, 
                                       metric_contexts, logger, init_writers);
  sampler_runner runner(model, args, writers, interrupt, logger);
  std::visit(runner, sampler_configs);
}

}  // namespace stan3

#endif  // STAN3_RUN_SAMPLERS_HPP

#ifndef STAN3_LOAD_SAMPLERS_HPP
#define STAN3_LOAD_SAMPLERS_HPP

#include <stan3/hmc_nuts_arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan3/metric_type.hpp>

#include <stan/callbacks/logger.hpp>
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

#include <boost/random/mixmax.hpp>
#include <tbb/parallel_for.h>

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace stan3 {

using rng_t = boost::random::mixmax;

/* Sampler configuration struct to hold initialized samplers and context */
template <typename SamplerType>
struct sampler_config {
  std::vector<SamplerType> samplers;
  std::vector<rng_t> rngs;
  std::vector<std::vector<double>> init_params;
};

/* Template specialization for different metric types */
template <metric_t MetricType>
struct sampler_traits {};

template <>
struct sampler_traits<metric_t::DIAG_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_diag_e_nuts<Model, rng_t>;
};

template <>
struct sampler_traits<metric_t::DENSE_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_dense_e_nuts<Model, rng_t>;
};

template <>
struct sampler_traits<metric_t::UNIT_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_unit_e_nuts<Model, rng_t>;
};

/* Convenience alias for the variant type */
template <typename Model>
using sampler_variant = std::variant<
  sampler_config<typename sampler_traits<metric_t::UNIT_E>::template sampler_type<Model>>,
  sampler_config<typename sampler_traits<metric_t::DIAG_E>::template sampler_type<Model>>,
  sampler_config<typename sampler_traits<metric_t::DENSE_E>::template sampler_type<Model>>
>;

/* Helper function to configure metric based on type */
template <metric_t MetricType, typename SamplerType, typename Model>
void configure_metric(SamplerType& sampler, const Model& model,
                     const stan::io::var_context* metric_context,
                     stan::callbacks::logger& logger) {
  if constexpr (MetricType == metric_t::DIAG_E) {
    Eigen::VectorXd inv_metric;
    try {
      if (metric_context) {
        inv_metric = stan::services::util::read_diag_inv_metric(
          *metric_context, model.num_params_r(), logger);
      } else {
        inv_metric = Eigen::VectorXd::Ones(model.num_params_r());
      }
    } catch (const std::exception& e) {
      // If reading fails, fall back to unit metric
      logger.info("Using unit diagonal metric (failed to read provided metric)");
      inv_metric = Eigen::VectorXd::Ones(model.num_params_r());
    }
    stan::services::util::validate_diag_inv_metric(inv_metric, logger);
    sampler.set_metric(inv_metric);
  } else if constexpr (MetricType == metric_t::DENSE_E) {
    Eigen::MatrixXd inv_metric;
    try {
      if (metric_context) {
        inv_metric = stan::services::util::read_dense_inv_metric(
          *metric_context, model.num_params_r(), logger);
      } else {
        inv_metric = Eigen::MatrixXd::Identity(model.num_params_r(), model.num_params_r());
      }
    } catch (const std::exception& e) {
      // If reading fails, fall back to identity matrix
      logger.info("Using identity matrix metric (failed to read provided metric)");
      inv_metric = Eigen::MatrixXd::Identity(model.num_params_r(), model.num_params_r());
    }
    stan::services::util::validate_dense_inv_metric(inv_metric, logger);
    sampler.set_metric(inv_metric);
  } else if constexpr (MetricType == metric_t::UNIT_E) {
    // Unit metric doesn't need explicit setting - it's the default
    // metric_context is ignored for unit metric
  }
}

/* Helper function to configure common sampler parameters */
template <typename SamplerType>
void configure_sampler_basic(SamplerType& sampler, 
                            const hmc_nuts_args& args) {
  sampler.set_nominal_stepsize(args.stepsize);
  sampler.set_stepsize_jitter(args.stepsize_jitter);
  sampler.set_max_depth(args.max_depth);
  
  sampler.get_stepsize_adaptation().set_mu(std::log(10 * args.stepsize));
  sampler.get_stepsize_adaptation().set_delta(args.delta);
  sampler.get_stepsize_adaptation().set_gamma(args.gamma);
  sampler.get_stepsize_adaptation().set_kappa(args.kappa);
  sampler.get_stepsize_adaptation().set_t0(args.t0);
}

/* Helper function to configure windowed adaptation (only for diag_e and dense_e) */
template <metric_t MetricType, typename SamplerType>
void configure_windowed_adaptation(SamplerType& sampler,
                                  const hmc_nuts_args& args,
                                  stan::callbacks::logger& logger) {
  if constexpr (MetricType == metric_t::DIAG_E || MetricType == metric_t::DENSE_E) {
    sampler.set_window_params(args.num_warmup, args.init_buffer, 
                             args.term_buffer, args.window, logger);
  }
  // Unit metric doesn't need windowed adaptation
}

/* Load and configure samplers for a specific metric type
 * 
 * @tparam MetricType The metric type enum value
 * @tparam Model The Stan model type
 * @param model Reference to the instantiated model
 * @param args HMC-NUTS arguments containing sampler configuration
 * @param init_contexts Vector of initialization contexts for each chain
 * @param metric_contexts Vector of metric contexts for each chain
 * @param logger Reference to logger for messages
 * @param init_writers Vector of writers for initialization output (can contain nullptrs)
 * @return sampler_config containing initialized samplers and related data
 */
template <metric_t MetricType, typename Model>
sampler_config<typename sampler_traits<MetricType>::template sampler_type<Model>>
load_samplers(Model& model,
              const hmc_nuts_args& args,
              const std::vector<std::shared_ptr<const stan::io::var_context>>& init_contexts,
              const std::vector<std::shared_ptr<const stan::io::var_context>>& metric_contexts,
              stan::callbacks::logger& logger,
              const std::vector<stan::callbacks::writer*>& init_writers) {
  
  using sampler_type = typename sampler_traits<MetricType>::template sampler_type<Model>;
  using config_type = sampler_config<sampler_type>;
  
  config_type config;
  config.samplers.reserve(args.num_chains);
  config.rngs.reserve(args.num_chains);
  config.init_params.reserve(args.num_chains);
  
  try {
    for (size_t i = 0; i < args.num_chains; ++i) {
      // Create RNG for this chain
      config.rngs.emplace_back(stan::services::util::create_rng(args.random_seed, i + 1));
      
      // Initialize parameters
      stan::io::var_context* init_context =
        const_cast<stan::io::var_context*>(init_contexts[i].get());
      
      // Handle case where init_writer might be null
      stan::callbacks::writer dummy_writer;
      stan::callbacks::writer* writer_to_use = 
        (init_writers[i] != nullptr) ? init_writers[i] : &dummy_writer;

      // initialize with timing message == false      
      auto init_params = stan::services::util::initialize(
        model, *init_context, config.rngs[i], args.init_radius, false, logger,  
        *writer_to_use);
      config.init_params.push_back(std::move(init_params));
      
      // Create and configure sampler
      config.samplers.emplace_back(model, config.rngs[i]);
      auto& sampler = config.samplers.back();
      
      // Configure metric
      configure_metric<MetricType>(sampler, model, metric_contexts[i].get(), logger);
      
      // Configure basic sampler parameters
      configure_sampler_basic(sampler, args);
      
      // Configure windowed adaptation (only for diag_e and dense_e)
      configure_windowed_adaptation<MetricType>(sampler, args, logger);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("Error configuring samplers: " + std::string(e.what()));
  }
  
  return config;
}

/* Factory function to create samplers based on runtime metric type
 * Returns a std::variant containing the appropriate sampler configuration
 */
template <typename Model>
sampler_variant<Model>
create_samplers(Model& model,
                const hmc_nuts_args& args,
                const std::vector<std::shared_ptr<const stan::io::var_context>>& init_contexts,
                const std::vector<std::shared_ptr<const stan::io::var_context>>& metric_contexts,
                stan::callbacks::logger& logger,
                const std::vector<stan::callbacks::writer*>& init_writers) {
  
  switch (args.metric_type) {
    case metric_t::UNIT_E:
      return load_samplers<metric_t::UNIT_E>(model, args, init_contexts, 
                                           metric_contexts, logger, init_writers);
    case metric_t::DIAG_E:
      return load_samplers<metric_t::DIAG_E>(model, args, init_contexts,
                                           metric_contexts, logger, init_writers);
    case metric_t::DENSE_E:
      return load_samplers<metric_t::DENSE_E>(model, args, init_contexts,
                                            metric_contexts, logger, init_writers);
    default:
      throw std::runtime_error("Unknown metric type: " + std::to_string(static_cast<int>(args.metric_type)));
  }
}

/* Helper visitor to run samplers regardless of their type */
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
    if (args_.num_chains == 1) {
      run_single_chain(config, 0);
    } else {
      run_multiple_chains(config);
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
      chain_idx + 1, args_.num_chains);
  }
  
  template <typename ConfigType>
  void run_multiple_chains(ConfigType& config) {
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, args_.num_chains, 1),
      [this, &config](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          run_single_chain(config, i);
        }
      },
      tbb::simple_partitioner());
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
  init_writers.reserve(args.num_chains);
  
  for (size_t i = 0; i < args.num_chains; ++i) {
    init_writers.push_back(
      writers[i].start_params_writer ? writers[i].start_params_writer.get() : nullptr);
  }
  auto sampler_configs = create_samplers(model, args, init_contexts, 
                                       metric_contexts, logger, init_writers);
  sampler_runner runner(model, args, writers, interrupt, logger);
  std::visit(runner, sampler_configs);
}

}  // namespace stan3

#endif  // STAN3_LOAD_SAMPLERS_HPP

#ifndef STAN3_ARGUMENTS_HPP
#define STAN3_ARGUMENTS_HPP

#include <CLI11/CLI11.hpp>
#include <stan3/algorithm_type.hpp>
#include <stan3/metric_type.hpp>
#include <string>
#include <map>
#include <memory>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif

namespace stan3 {

/* Base model loading arguments */
struct model_args {
  unsigned int random_seed = 1;
  std::string data_file;
};

/* Specify, record valid initial parameter values */
struct init_args {
  double init_radius = 2.0;
  std::vector<std::string> init_files;
};

/* Combined arguments for inference operations */
struct inference_args {
  model_args model;
  size_t num_chains = 1;
  init_args init;
  std::string output_dir;
};

/* HMC-NUTS specific arguments */
struct hmc_nuts_args {
  inference_args base;

  int num_warmup = 1000;
  int num_samples = 1000;
  int thin = 1;
  int refresh = 100;
  metric_t metric_type = metric_t::DIAG_E;
  std::vector<std::string> metric_files;
  double stepsize = 1.0;
  double stepsize_jitter = 0.0;
  int max_depth = 10;

  // HMC output options
  bool save_start_params = false;
  bool save_warmup = false;
  bool save_diagnostics = false;
  bool save_metric = false;
  
  // NUTS adaptation options
  double delta = 0.8;
  double gamma = 0.05;
  double kappa = 0.75;
  double t0 = 10.0;
  unsigned int init_buffer = 75;
  unsigned int term_buffer = 50;
  unsigned int window = 25;
};

/* Custom validator for JSON input files */
struct JSONFileValidator : public CLI::Validator {
  JSONFileValidator() {
    name_ = "JSONFile";
    func_ = [](const std::string& str) -> std::string {
      if (str.empty()) {
        return std::string{};
      }
      if (!std::filesystem::exists(str)) {
        return "JSON file does not exist: " + str;
      }
      std::ifstream test_stream(str);
      if (!test_stream.good()) {
        return "JSON file is not readable (permission denied?): " + str;
      }
      char first_char;
      test_stream >> std::ws >> first_char;
      if (first_char != '{') {
        return "File must contain a JSON object (starting with '{'): " + str;
      }
      return std::string{};
    };
  }
};

/* Custom validator for vector of JSON files */
struct JSONFileVectorValidator : public CLI::Validator {
  JSONFileVectorValidator() {
    name_ = "JSONFileVector";
    func_ = [](const std::string& str) -> std::string {
      if (str.empty()) {
        return std::string{};
      }
      
      JSONFileValidator single_validator;
      return single_validator(str);
    };
  }
};

/* Function to create string-to-enum mapping for metric type */
inline std::map<std::string, metric_t> create_metric_map() {
  return {
    {"unit_e", metric_t::UNIT_E},
    {"diag_e", metric_t::DIAG_E},
    {"dense_e", metric_t::DENSE_E}
  };
}

/* Function to create a unique temporary directory */
inline std::string create_temp_output_dir() {
  auto temp_base = std::filesystem::temp_directory_path();
  
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto pid = std::to_string(getpid());

  std::stringstream ss;
  ss << "stan3_output_" << time_t << "_" << pid;
  auto temp_dir = temp_base / ss.str();
  std::filesystem::create_directories(temp_dir);
  
  return temp_dir.string();
}

/* Function to clean up temporary directory (call at program exit) */
inline void cleanup_temp_dir(const std::string& dir_path) {
  if (dir_path.find("stan3_output_") != std::string::npos) {
    std::error_code ec;
    std::filesystem::remove_all(dir_path, ec);
    if (ec) {
      std::cerr << "Warning: Could not clean up temporary directory: " 
                << dir_path << " (" << ec.message() << ")" << std::endl;
    }
  }
}

/* Function to setup model loading options only */
inline void setup_model_options(CLI::App& app, model_args& args) {
  app.add_option("--seed", args.random_seed, 
                 "Random seed for initialization")
    ->capture_default_str();
  
  app.add_option("--data", args.data_file, 
                 "Data inputs file")
    ->check(JSONFileValidator{});
}

/* Function to setup initialization options */
inline void setup_init_options(CLI::App& app, init_args& args) {
  app.add_option("--init-radius", args.init_radius, 
                 "Initial radius for parameter initialization")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  app.add_option("--inits", args.init_files, 
                 "Initial parameter values. "
                 "Comma-separated for multiple files or repeat option for per-chain files.")
    ->check(JSONFileVectorValidator{});
}

/* Function to setup inference options */
inline void setup_inference_options(CLI::App& app, inference_args& args) {
  app.add_option("--chains", args.num_chains, 
                 "Number of inference chains to run")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();

  app.add_option("-o,--output-dir", args.output_dir, 
                 "Directory for all output files")
    ->default_function([]() { return create_temp_output_dir(); })
    ->capture_default_str();
}


/* Standalone parser for model loading */
inline bool parse_model_args(int argc, char** argv, model_args& args, std::string& error_msg) {
  CLI::App app{"Stan3 Model Loader"};
  setup_model_options(app, args);
  
  try {
    app.parse(argc, argv);
    return true;
  } catch (const CLI::ParseError& e) {
    error_msg = "Model argument parsing failed: " + std::to_string(e.get_exit_code());
    return false;
  }
}

/* Standalone parser for inference arguments */
inline bool parse_inference_args(int argc, char** argv, inference_args& args, std::string& error_msg) {
  CLI::App app{"Stan3 Inference Setup"};
  setup_model_options(app, args.model);
  setup_init_options(app, args.init);
  setup_inference_options(app, args);
  
  try {
    app.parse(argc, argv);
    return true;
  } catch (const CLI::ParseError& e) {
    error_msg = "Inference argument parsing failed: " + std::to_string(e.get_exit_code());
    return false;
  }
}

/* Function for HMC-NUTS specific validation */
inline bool validate_hmc_arguments(const hmc_nuts_args& args, std::string& error_message) {
  if (args.num_samples > 0 && args.thin > args.num_samples) {
    error_message = "Error: thin (" + std::to_string(args.thin) + 
      ") cannot exceed --samples (" + 
      std::to_string(args.num_samples) + ")";
    return false;
  }
  
  // Validate init_files: must be empty, size 1, or size num_chains
  if (!args.base.init.init_files.empty() && 
      args.base.init.init_files.size() != 1 && 
      args.base.init.init_files.size() != args.base.num_chains) {
    error_message = "Error: --inits must specify either 1 file (for all chains) or " +
                   std::to_string(args.base.num_chains) + " files (one per chain). " +
                   "Found " + std::to_string(args.base.init.init_files.size()) + " files.";
    return false;
  }
  
  // Validate metric_files: must be empty, size 1, or size num_chains
  if (!args.metric_files.empty() && 
      args.metric_files.size() != 1 && 
      args.metric_files.size() != args.base.num_chains) {
    error_message = "Error: --metric must specify either 1 file (for all chains) or " +
                   std::to_string(args.base.num_chains) + " files (one per chain). " +
                   "Found " + std::to_string(args.metric_files.size()) + " files.";
    return false;
  }
  
  return true;
}

/* Standalone parser for HMC arguments (includes inference args) */
inline bool parse_hmc_args(int argc, char** argv, hmc_nuts_args& args, std::string& error_msg) {
  CLI::App app{"Stan3 HMC Sampler"};
  setup_model_options(app, args.base.model);
  setup_init_options(app, args.base.init);
  
  // Add HMC-specific options
  auto hmc_opts = app.add_option_group("HMC Options");
  auto nuts_opts = app.add_option_group("NUTS Adaptation Options");
  auto output_opts = app.add_option_group("Output Options");
  
  // Copy the HMC option setup from setup_hmc_subcommand but directly to app
  auto metric_map = create_metric_map();
  hmc_opts->add_option("--metric-type", args.metric_type, 
                       "Type of metric to use in Hamiltonian dynamics")
    ->transform(CLI::CheckedTransformer(metric_map, CLI::ignore_case))
    ->capture_default_str();
  

  hmc_opts->add_option("--metric", args.metric_files, 
                       "Precomputed inverse metric. "
                       "Comma-separated for multiple files or repeat option for per-chain files.")
    ->check(JSONFileVectorValidator{});
  
  hmc_opts->add_option("--stepsize", args.stepsize, 
                       "Step size for discrete evolution")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--jitter", args.stepsize_jitter, 
                       "Uniformly random jitter of the stepsize, in percent")
    ->check(CLI::Range(0.0, 1.0))
    ->capture_default_str();
  
  hmc_opts->add_option("--max-depth", args.max_depth, 
                       "Maximum tree depth")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--warmup", args.num_warmup, 
                       "Number of warmup iterations")
    ->check(CLI::NonNegativeNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--samples", args.num_samples, 
                       "Number of sampling iterations")
    ->check(CLI::NonNegativeNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--thin", args.thin, 
                       "Period between saved samples")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--refresh", args.refresh, 
                       "Number of iterations between progress messages")
    ->capture_default_str();
  
  // NUTS adaptation options
  nuts_opts->add_option("--delta", args.delta, 
                        "Adaptation target acceptance statistic")
    ->check(CLI::Range(0.0, 1.0))
    ->capture_default_str();
  
  nuts_opts->add_option("--gamma", args.gamma, 
                        "Adaptation regularization scale")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--kappa", args.kappa, 
                        "Adaptation relaxation exponent")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--t0", args.t0, 
                        "Adaptation iteration offset")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--init-buffer", args.init_buffer, 
                        "Width of initial fast adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--term-buffer", args.term_buffer, 
                        "Width of final fast adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--window", args.window, 
                        "Initial width of slow adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  // Output options
  output_opts->add_flag("--save-inits", args.save_start_params,
                        "Save initial parameter values?")
    ->capture_default_str();

  output_opts->add_flag("--save-warmup", args.save_warmup, 
                        "Save warmup iterations?")
    ->capture_default_str();

  output_opts->add_flag("--save-metric", args.save_metric, 
                        "Save adapted metric?")
    ->capture_default_str();

  output_opts->add_flag("--save-diag", args.save_diagnostics, 
                        "Save unconstrained parameter values and gradients?")
    ->capture_default_str();
  
  try {
    app.parse(argc, argv);
    
    // Validate HMC-specific constraints
    std::string hmc_error;
    if (!validate_hmc_arguments(args, hmc_error)) {
      error_msg = hmc_error;
      return false;
    }
    
    return true;
  } catch (const CLI::ParseError& e) {
    error_msg = "HMC argument parsing failed: " + std::to_string(e.get_exit_code());
    return false;
  }
}

/* Simplified backward-compatible CLI setup for main.cpp */
inline CLI::App* setup_backward_compatible_cli(CLI::App& app, hmc_nuts_args& hmc_args) {
  app.description("Stan3 - Command line interface for Stan");
  app.require_subcommand(1);
  
  // Create HMC subcommand
  auto hmc_sub = app.add_subcommand("hmc", "Hamiltonian Monte Carlo with NUTS");
  
  // Setup all options directly on the HMC subcommand
  setup_model_options(*hmc_sub, hmc_args.base.model);
  setup_init_options(*hmc_sub, hmc_args.base.init);
  setup_inference_options(*hmc_sub, hmc_args.base);
  
  // Add HMC-specific options (copy from parse_hmc_args)
  auto hmc_opts = hmc_sub->add_option_group("HMC Options");
  auto nuts_opts = hmc_sub->add_option_group("NUTS Adaptation Options");
  auto output_opts = hmc_sub->add_option_group("Output Options");
  
  auto metric_map = create_metric_map();
  hmc_opts->add_option("--metric-type", hmc_args.metric_type, 
                       "Type of metric to use in Hamiltonian dynamics")
    ->transform(CLI::CheckedTransformer(metric_map, CLI::ignore_case))
    ->capture_default_str();
  
  hmc_opts->add_option("--metric", hmc_args.metric_files, 
                       "Precomputed inverse metric")
    ->check(JSONFileVectorValidator{});
  
  hmc_opts->add_option("--stepsize", hmc_args.stepsize, 
                       "Step size for discrete evolution")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--jitter", hmc_args.stepsize_jitter, 
                       "Uniformly random jitter of the stepsize, in percent")
    ->check(CLI::Range(0.0, 1.0))
    ->capture_default_str();
  
  hmc_opts->add_option("--max-depth", hmc_args.max_depth, 
                       "Maximum tree depth")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--warmup", hmc_args.num_warmup, 
                       "Number of warmup iterations")
    ->check(CLI::NonNegativeNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--samples", hmc_args.num_samples, 
                       "Number of sampling iterations")
    ->check(CLI::NonNegativeNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--thin", hmc_args.thin, 
                       "Period between saved samples")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  hmc_opts->add_option("--refresh", hmc_args.refresh, 
                       "Number of iterations between progress messages")
    ->capture_default_str();
  
  // NUTS adaptation options
  nuts_opts->add_option("--delta", hmc_args.delta, 
                        "Adaptation target acceptance statistic")
    ->check(CLI::Range(0.0, 1.0))
    ->capture_default_str();
  
  nuts_opts->add_option("--gamma", hmc_args.gamma, 
                        "Adaptation regularization scale")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--kappa", hmc_args.kappa, 
                        "Adaptation relaxation exponent")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--t0", hmc_args.t0, 
                        "Adaptation iteration offset")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--init-buffer", hmc_args.init_buffer, 
                        "Width of initial fast adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--term-buffer", hmc_args.term_buffer, 
                        "Width of final fast adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  nuts_opts->add_option("--window", hmc_args.window, 
                        "Initial width of slow adaptation interval")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  // Output options
  output_opts->add_flag("--save-inits", hmc_args.save_start_params,
                        "Save initial parameter values?")
    ->capture_default_str();

  output_opts->add_flag("--save-warmup", hmc_args.save_warmup, 
                        "Save warmup iterations?")
    ->capture_default_str();

  output_opts->add_flag("--save-metric", hmc_args.save_metric, 
                        "Save adapted metric?")
    ->capture_default_str();

  output_opts->add_flag("--save-diag", hmc_args.save_diagnostics, 
                        "Save unconstrained parameter values and gradients?")
    ->capture_default_str();
  
  return hmc_sub;
}

/* Function to finalize arguments after CLI parsing */
inline void finalize_hmc_arguments(hmc_nuts_args& args) {
  if (args.base.output_dir.empty()) {
    args.base.output_dir = create_temp_output_dir();
    std::cout << "created tmp dir " << args.base.output_dir << std::endl;
  }
}

/* Helper function to get init file for a specific chain */
inline std::string get_init_file_for_chain(const init_args& args, size_t chain_idx) {
  if (args.init_files.empty()) {
    return "";
  }
  if (args.init_files.size() == 1) {
    return args.init_files[0];
  }
  return args.init_files[chain_idx];
}

/* Helper function to get metric file for a specific chain (HMC-specific) */
inline std::string get_metric_file_for_chain(const hmc_nuts_args& args, size_t chain_idx) {
  if (args.metric_files.empty()) {
    return "";
  }
  if (args.metric_files.size() == 1) {
    return args.metric_files[0];
  }
  return args.metric_files[chain_idx];
}

}  // namespace stan3

#endif  // STAN3_ARGUMENTS_HPP

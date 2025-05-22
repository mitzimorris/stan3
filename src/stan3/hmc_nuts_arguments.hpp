#ifndef STAN3_HMC_NUTS_ARGUMENTS_HPP
#define STAN3_HMC_NUTS_ARGUMENTS_HPP

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

/* Command line arguments */
struct hmc_nuts_args {
  // Algorithm options
  algorithm_t algorithm = algorithm_t::STAN2_HMC;
  size_t num_chains = 1;
  unsigned int random_seed = 1;
  
  // Model options
  double init_radius = 2.0;
  std::string data_file;
  std::vector<std::string> init_files;
  
  // HMC options
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
  std::string output_dir;
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

/* Custom validator for JSON input files
 * If no filename specified, set to arg value to empty string,
 * else check that file exists, is readable, and has correct opening '{'.
 */
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

/* Function to create string-to-enum mapping for algorithm type */
inline std::map<std::string, algorithm_t> create_algorithm_map() {
  return {
    {"hmc", algorithm_t::STAN2_HMC},
    {"mle", algorithm_t::MLE},
    {"pathfinder", algorithm_t::PATHFINDER},
    {"advi", algorithm_t::ADVI},
    {"gq", algorithm_t::STANDALONE_GQ}
  };
}

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

/* Function to setup CLI options */
inline void setup_cli(CLI::App& app, hmc_nuts_args& args) {
  auto algorithm_opts = app.add_option_group("Algorithm Options");
  auto model_inits_opts = app.add_option_group("Model Options");
  auto hmc_opts = app.add_option_group("HMC Options");
  auto nuts_opts = app.add_option_group("NUTS Adaptation Options");
  auto output_opts = app.add_option_group("Output Options");
  
  // Algorithm options
  auto algorithm_map = create_algorithm_map();
  algorithm_opts->add_option("--algorithm", args.algorithm, 
                             "Inference algorithm to run")
    ->transform(CLI::CheckedTransformer(algorithm_map, CLI::ignore_case))
    ->capture_default_str();
  
  algorithm_opts->add_option("--chains", args.num_chains, 
                             "Number of Markov chains to run")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  algorithm_opts->add_option("--seed", args.random_seed, 
                             "Random seed for initialization")
    ->capture_default_str();
  
  // Model options
  model_inits_opts->add_option("--data", args.data_file, 
                               "Data inputs file")
    ->check(JSONFileValidator{});
  
  model_inits_opts->add_option("--init-radius", args.init_radius, 
                               "Initial radius for parameter initialization")
    ->check(CLI::PositiveNumber)
    ->capture_default_str();
  
  model_inits_opts->add_option("--inits", args.init_files, 
                               "Initial parameter values. "
                               "Comma-separated for multiple files or repeat option for per-chain files.")
    ->check(JSONFileVectorValidator{});

  // HMC options
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
  output_opts->add_option("-o,--output-dir", args.output_dir, 
                          "Output directory for samples")
    ->default_function([]() { return create_temp_output_dir(); })
    ->capture_default_str();
  
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
}

/* Function for additional validation beyond CLI11 capabilities */
inline bool validate_arguments(const hmc_nuts_args& args, std::string& error_message) {
  if (args.thin > args.num_samples) {
    error_message = "Error: thin (" + std::to_string(args.thin) + 
      ") cannot exceed --samples (" + 
      std::to_string(args.num_samples) + ")";
    return false;
  }
  
  // Validate init_files: must be empty, size 1, or size num_chains
  if (!args.init_files.empty() && 
      args.init_files.size() != 1 && 
      args.init_files.size() != args.num_chains) {
    error_message = "Error: --inits must specify either 1 file (for all chains) or " +
                   std::to_string(args.num_chains) + " files (one per chain). " +
                   "Found " + std::to_string(args.init_files.size()) + " files.";
    return false;
  }
  
  // Validate metric_files: must be empty, size 1, or size num_chains
  if (!args.metric_files.empty() && 
      args.metric_files.size() != 1 && 
      args.metric_files.size() != args.num_chains) {
    error_message = "Error: --metric must specify either 1 file (for all chains) or " +
                   std::to_string(args.num_chains) + " files (one per chain). " +
                   "Found " + std::to_string(args.metric_files.size()) + " files.";
    return false;
  }
  
  return true;
}

/* Function to finalize arguments after CLI parsing */
inline void finalize_arguments(hmc_nuts_args& args) {
  if (args.output_dir.empty()) {
    args.output_dir = create_temp_output_dir();
  }
}

/* Helper function to get init file for a specific chain */
inline std::string get_init_file_for_chain(const hmc_nuts_args& args, size_t chain_idx) {
  if (args.init_files.empty()) {
    return "";
  }
  if (args.init_files.size() == 1) {
    return args.init_files[0];
  }
  return args.init_files[chain_idx];
}

/* Helper function to get metric file for a specific chain */
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

#endif  // STAN3_HMC_NUTS_ARGUMENTS_HPP

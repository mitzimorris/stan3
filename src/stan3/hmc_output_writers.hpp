#ifndef STAN3_HMC_OUTPUT_WRITERS_HPP
#define STAN3_HMC_OUTPUT_WRITERS_HPP

#include <stan3/output_writers.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>

namespace stan3 {

/* Container for all output writers needed for a single HMC-NUTS chain */
struct hmc_nuts_writers {
  std::unique_ptr<csv_writer> sample_writer;
  std::unique_ptr<csv_writer> start_params_writer;
  std::unique_ptr<csv_writer> diagnostics_writer;
  std::unique_ptr<json_writer> metric_writer;
};

/* Create HMC-NUTS output writers for a single chain
 * 
 * @param args HMC-NUTS arguments containing output configuration
 * @param model_name Name of the Stan model
 * @param timestamp Timestamp string
 * @param chain_id Chain number (1-indexed)
 * @param comment_prefix Optional comment prefix for CSV files
 * @return hmc_nuts_writers struct containing all writers for the chain
 */
hmc_nuts_writers create_hmc_nuts_single_chain_writers(
    const hmc_nuts_args& args,
    const std::string& model_name,
    const std::string& timestamp,
    unsigned int chain_id,
    const std::string& comment_prefix = "#") {
    
  hmc_nuts_writers writers;
  
  // Sample writer is always required
  writers.sample_writer = create_writer<csv_writer>(
      args.output_dir, model_name, timestamp, chain_id, 
      "sample", ".csv", comment_prefix);
  
  // Optional writers
  if (args.save_start_params) {
    writers.start_params_writer = create_writer<csv_writer>(
        args.output_dir, model_name, timestamp, chain_id,
        "start_params", ".csv");
  } else {
    writers.start_params_writer = nullptr;
  }
  
  if (args.save_diagnostics) {
    writers.diagnostics_writer = create_writer<csv_writer>(
        args.output_dir, model_name, timestamp, chain_id,
        "param_grads", ".csv", comment_prefix);
  } else {
    writers.diagnostics_writer = nullptr;
  }
  
  if (args.save_metric) {
    writers.metric_writer = create_writer<json_writer>(
        args.output_dir, model_name, timestamp, chain_id,
        "metric", ".json");
  } else {
    writers.metric_writer = nullptr;
  }
  
  return writers;
}

/* Create HMC-NUTS output writers for a multi-chain run.
 * 
 * @param args HMC-NUTS arguments containing output configuration
 * @param model_name Name of the Stan model
 * @param comment_prefix Optional comment prefix for CSV files
 * @return Vector of hmc_nuts_writers, one for each chain
 */
std::vector<hmc_nuts_writers> create_hmc_nuts_multi_chain_writers(
    const hmc_nuts_args& args,
    const std::string& model_name,
    const std::string& comment_prefix = "#") {
    
  ensure_output_directory(args.output_dir);
  std::string timestamp = generate_timestamp();
  
  std::vector<hmc_nuts_writers> multi_writers;
  multi_writers.reserve(args.num_chains);
  
  for (unsigned int i = 1; i <= args.num_chains; ++i) {
    multi_writers.push_back(create_hmc_nuts_single_chain_writers(
        args, model_name, timestamp, i, comment_prefix));
  }
  
  return multi_writers;
}

}  // namespace stan3

#endif  // STAN3_HMC_OUTPUT_WRITERS_HPP

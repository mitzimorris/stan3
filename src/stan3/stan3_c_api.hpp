#ifndef STAN3_C_API_HPP
#define STAN3_C_API_HPP

#include <stan3/arguments.hpp>
#include <stan3/load_model.hpp>
#include <stan3/run_hmc_nuts.hpp>
#include <stan/model/model_base.hpp>

#include <memory>
#include <string>
#include <cstring>

namespace stan3 {
namespace c_api {

/* Internal model storage */
extern std::unique_ptr<stan::model::model_base> g_model;
extern std::string g_last_error;

/* Helper function to safely copy error message to C buffer
 * 
 * @param error_msg C++ error message string
 * @param buffer C buffer to copy to
 * @param buffer_size Size of C buffer
 */
inline void copy_error_message(const std::string& error_msg, 
                              char* buffer, size_t buffer_size) {
  if (buffer && buffer_size > 0) {
    size_t copy_len = std::min(error_msg.length(), buffer_size - 1);
    std::strncpy(buffer, error_msg.c_str(), copy_len);
    buffer[copy_len] = '\0';
  }
}

/* Helper function to convert argc/argv to vector<string>
 * 
 * @param argc Number of arguments
 * @param argv Array of argument strings
 * @return Vector of strings
 */
inline std::vector<std::string> argv_to_vector(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return args;
}

/* Helper function to convert vector<string> back to argc/argv format
 * for passing to CLI11 parsers
 * 
 * @param args Vector of argument strings
 * @param argc_out Output parameter for argc
 * @param argv_out Output parameter for argv (caller must free)
 */
inline void vector_to_argv(const std::vector<std::string>& args,
                          int& argc_out, char**& argv_out) {
  argc_out = static_cast<int>(args.size());
  argv_out = new char*[argc_out];
  for (int i = 0; i < argc_out; ++i) {
    argv_out[i] = new char[args[i].length() + 1];
    std::strcpy(argv_out[i], args[i].c_str());
  }
}

/* Helper function to free argv created by vector_to_argv
 * 
 * @param argc Number of arguments
 * @param argv Array to free
 */
inline void free_argv(int argc, char** argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

/* Internal implementation of model loading
 * 
 * @param argc Number of arguments
 * @param argv Array of argument strings
 * @param error_msg Output parameter for error message
 * @return Success flag
 */
bool load_model_impl(int argc, char** argv, std::string& error_msg);

/* Internal implementation of sampler running
 * 
 * @param argc Number of arguments
 * @param argv Array of argument strings  
 * @param error_msg Output parameter for error message
 * @return Success flag
 */
bool run_samplers_impl(int argc, char** argv, std::string& error_msg);

}  // namespace c_api
}  // namespace stan3

#endif  /* STAN3_C_API_HPP */

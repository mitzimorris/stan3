#define STAN3_BUILDING_LIBRARY
#include "stan3_c_api.h"
#include "stan3_c_api.hpp"

#include <stan3/arguments.hpp>
#include <stan3/load_model.hpp>
#include <stan3/run_hmc_nuts.hpp>

#include <stdexcept>
#include <iostream>

namespace stan3 {
namespace c_api {

/* Static storage for the loaded model and error messages */
std::unique_ptr<stan::model::model_base> g_model = nullptr;
std::string g_last_error;

bool load_model_impl(int argc, char** argv, std::string& error_msg) {
  try {
    stan3::model_args args;
    
    if (!stan3::parse_model_args(argc, argv, args, error_msg)) {
      return false;
    }
    
    auto& model = stan3::load_model(args);
    g_model = std::unique_ptr<stan::model::model_base>(&model);
    
    return true;
    
  } catch (const std::invalid_argument& e) {
    error_msg = "Invalid argument: " + std::string(e.what());
    return false;
  } catch (const std::runtime_error& e) {
    error_msg = "Runtime error: " + std::string(e.what());
    return false;
  } catch (const std::exception& e) {
    error_msg = "Error loading model: " + std::string(e.what());
    return false;
  } catch (...) {
    error_msg = "Unknown error occurred while loading model";
    return false;
  }
}

bool run_samplers_impl(int argc, char** argv, std::string& error_msg) {
  try {
    if (!g_model) {
      error_msg = "No model loaded. Call stan3_load_model() first.";
      return false;
    }
    
    stan3::hmc_nuts_args args;
    
    if (!stan3::parse_hmc_args(argc, argv, args, error_msg)) {
      return false;
    }
    
    int result = stan3::run_hmc(args, *g_model);
    if (result != 0) {
      error_msg = "Sampling failed with exit code: " + std::to_string(result);
      return false;
    }
    
    return true;
    
  } catch (const std::invalid_argument& e) {
    error_msg = "Invalid argument: " + std::string(e.what());
    return false;
  } catch (const std::runtime_error& e) {
    error_msg = "Runtime error: " + std::string(e.what());
    return false;
  } catch (const std::exception& e) {
    error_msg = "Error running samplers: " + std::string(e.what());
    return false;
  } catch (...) {
    error_msg = "Unknown error occurred while running samplers";
    return false;
  }
}

}  // namespace c_api
}  // namespace stan3

/* C API Implementation */
extern "C" {

STAN3_API int stan3_load_model(int argc, char** argv, 
                               char* error_message, size_t error_message_size) {
  if (argc < 0 || !argv) {
    stan3::c_api::g_last_error = "Invalid arguments: argc < 0 or argv is NULL";
    stan3::c_api::copy_error_message(stan3::c_api::g_last_error, 
                                    error_message, error_message_size);
    return STAN3_ERROR_INVALID_ARGS;
  }
  
  std::string error_msg;
  bool success = stan3::c_api::load_model_impl(argc, argv, error_msg);
  
  if (!success) {
    stan3::c_api::g_last_error = error_msg;
    stan3::c_api::copy_error_message(error_msg, error_message, error_message_size);
    
    if (error_msg.find("parsing failed") != std::string::npos) {
      return STAN3_ERROR_PARSING;
    } else if (error_msg.find("Invalid argument") != std::string::npos) {
      return STAN3_ERROR_INVALID_ARGS;
    } else {
      return STAN3_ERROR_MODEL_LOAD;
    }
  }
  
  stan3::c_api::g_last_error.clear();
  if (error_message && error_message_size > 0) {
    error_message[0] = '\0';
  }
  
  return STAN3_SUCCESS;
}

STAN3_API int stan3_run_samplers(int argc, char** argv,
                                 char* error_message, size_t error_message_size) {
  if (argc < 0 || !argv) {
    stan3::c_api::g_last_error = "Invalid arguments: argc < 0 or argv is NULL";
    stan3::c_api::copy_error_message(stan3::c_api::g_last_error, 
                                    error_message, error_message_size);
    return STAN3_ERROR_INVALID_ARGS;
  }
  
  std::string error_msg;
  bool success = stan3::c_api::run_samplers_impl(argc, argv, error_msg);
  
  if (!success) {
    stan3::c_api::g_last_error = error_msg;
    stan3::c_api::copy_error_message(error_msg, error_message, error_message_size);
    
    if (error_msg.find("No model loaded") != std::string::npos) {
      return STAN3_ERROR_MODEL_LOAD;
    } else if (error_msg.find("parsing failed") != std::string::npos) {
      return STAN3_ERROR_PARSING;
    } else if (error_msg.find("Invalid argument") != std::string::npos) {
      return STAN3_ERROR_INVALID_ARGS;
    } else {
      return STAN3_ERROR_SAMPLING;
    }
  }
  
  stan3::c_api::g_last_error.clear();
  if (error_message && error_message_size > 0) {
    error_message[0] = '\0';
  }
  
  return STAN3_SUCCESS;
}

STAN3_API const char* stan3_get_model_name(void) {
  if (!stan3::c_api::g_model) {
    return NULL;
  }
  
  try {
    static std::string model_name = stan3::c_api::g_model->model_name();
    return model_name.c_str();
  } catch (...) {
    return NULL;
  }
}

STAN3_API int stan3_is_model_loaded(void) {
  return stan3::c_api::g_model ? 1 : 0;
}

STAN3_API const char* stan3_get_last_error(void) {
  if (stan3::c_api::g_last_error.empty()) {
    return NULL;
  }
  return stan3::c_api::g_last_error.c_str();
}

STAN3_API void stan3_clear_error(void) {
  stan3::c_api::g_last_error.clear();
}

}  /* extern "C" */

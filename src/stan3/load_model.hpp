#ifndef STAN3_LOAD_MODEL_HPP
#define STAN3_LOAD_MODEL_HPP

#include <stan/model/model_base.hpp>
#include <boost/random/mixmax.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>


// Forward declaration for model creation function
// This is defined in each compiled Stan model
extern stan::model::model_base& new_model(stan::io::var_context& data_context, 
                                  unsigned int seed,
                                  std::ostream* msg_stream);

namespace stan3 {

/**
 * Instantiates a model given data and a specified random seed.
 * 
 * @param config_alg Algorithm configuration containing random seed
 * @param config_model Model configuration containing input data
 * @return instantiated model
 */
std::unique_ptr<stan::model::model_base>
load_model(stan::io::var_context& data_context,
	   unsigned int seed) {
  std::stringstream err_msg;
  auto& model_ref = ::new_model(data_context, seed, &err_msg);
  auto model_ptr = &model_ref;
  if (!err_msg.str().empty()) {
    throw std::runtime_error("Error in new_model: " + err_msg.str());
  } 
  return std::unique_ptr<stan::model::model_base>(model_ptr);
}

}  // namespace stan3

#endif  // STAN3_LOAD_MODEL_HPP

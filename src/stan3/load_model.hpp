#ifndef STAN3_LOAD_MODEL_HPP
#define STAN3_LOAD_MODEL_HPP

#include <stan3/arguments.hpp>
#include <stan3/read_json_data.hpp>
#include <stan/io/var_context.hpp>
#include <stan/model/model_base.hpp>

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

/* Load and instantiate a Stan model using the provided arguments
 * 
 * @param args Base arguments containing data file and random seed
 * @return Reference to the instantiated model
 * @throws std::invalid_argument if data file cannot be read
 * @throws std::runtime_error if model instantiation fails
 */
stan::model::model_base&
load_model(const model_args& args) {
  std::stringstream err_msg;

  std::shared_ptr<const stan::io::var_context> data_context;
  try {
    data_context = stan3::read_json_data(args.data_file);
  } catch (const std::exception &e) {
    err_msg << "Error reading input data, "
	    << e.what() << std::endl;
    throw std::invalid_argument(err_msg.str());
  }
  stan::io::var_context* raw_context = const_cast<stan::io::var_context*>(data_context.get());
  auto& model = ::new_model(*raw_context, args.random_seed, &err_msg);
  if (!err_msg.str().empty()) {
    throw std::runtime_error("Error in new_model: " + err_msg.str());
  }
  return model;
}

}  // namespace stan3
#endif  // STAN3_LOAD_MODEL_HPP

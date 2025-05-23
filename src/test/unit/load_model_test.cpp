#include <stan3/load_model.hpp>
#include <stan3/read_json_data.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/bernoulli.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

#include <gtest/gtest.h>

TEST(LoadModelTest, LoadModelWithValidData) {
  auto data_context = stan3::read_json_data("src/test/test-models/bernoulli.data.json");
  EXPECT_NO_THROW({
    auto model = stan3::load_model(*data_context, 12345);
    EXPECT_TRUE(model != nullptr);
    EXPECT_FALSE(model->model_name().empty());
  });
}

TEST(LoadModelTest, LoadModelWithEmptyData) {
  stan::io::empty_var_context empty_context;
  EXPECT_THROW(stan3::load_model(empty_context, 12345), std::runtime_error);
}

TEST(LoadModelTest, LoadModelParameterCount) {
  auto data_context = stan3::read_json_data("src/test/test-models/bernoulli.data.json");
  auto model = stan3::load_model(*data_context, 12345);
  EXPECT_EQ(model->num_params_r(), 1);
  
  std::vector<std::string> param_names;
  model->constrained_param_names(param_names, false, false);
  EXPECT_FALSE(param_names.empty());
  EXPECT_EQ(param_names[0], "theta");
  
  std::vector<std::string> uparam_names;
  model->unconstrained_param_names(uparam_names, false, false);
  EXPECT_FALSE(uparam_names.empty());
  EXPECT_EQ(uparam_names[0], "theta");
}

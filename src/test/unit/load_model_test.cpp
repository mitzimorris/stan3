#include <stan3/load_model.hpp>
#include <stan3/arguments.hpp>

#include <test/test-models/bernoulli.hpp>

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

TEST(LoadModelTest, LoadModelWithValidData) {
  stan3::stan3_args args;
  args.data_file = "src/test/test-models/bernoulli.data.json";
  args.random_seed = 12345;

  auto& model = stan3::load_model(args);

  EXPECT_FALSE(model.model_name().empty());
  EXPECT_EQ(model.model_name(), "bernoulli_model");
  EXPECT_EQ(model.num_params_r(), 1);
  
  std::vector<std::string> param_names;
  model.constrained_param_names(param_names, false, false);
  EXPECT_FALSE(param_names.empty());
  EXPECT_EQ(param_names[0], "theta");
  
  std::vector<std::string> uparam_names;
  model.unconstrained_param_names(uparam_names, false, false);
  EXPECT_FALSE(uparam_names.empty());
  EXPECT_EQ(uparam_names[0], "theta");
}

TEST(LoadModelTest, LoadModelWithNonexistentFile) {
  stan3::stan3_args args;
  args.data_file = "nonexistent_file.json";
  args.random_seed = 12345;
  
  EXPECT_THROW({
    stan3::load_model(args);
  }, std::invalid_argument);
}

TEST(LoadModelTest, LoadModelWithInvalidJSON) {
  stan3::stan3_args args;
  args.data_file = "src/test/json/invalid_data.json";
  args.random_seed = 12345;
  
  EXPECT_THROW({
    stan3::load_model(args);
  }, std::invalid_argument);
}

TEST(LoadModelTest, LoadModelWithHmcArgs) {
  // Test that function works with derived HMC args via inheritance
  stan3::hmc_nuts_args hmc_args;
  hmc_args.data_file = "src/test/test-models/bernoulli.data.json";
  hmc_args.random_seed = 54321;
  
  // Should work via polymorphism
  auto& model = stan3::load_model(hmc_args);
  
  EXPECT_FALSE(model.model_name().empty());
  EXPECT_EQ(model.model_name(), "bernoulli_model");
}


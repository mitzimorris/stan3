#include <stan3/arguments.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

/* Test base stan3_args functionality */
TEST(Stan3ArgsTest, GetInitFileForChain_EmptyFiles) {
  stan3::stan3_args args;
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 5), "");
}

TEST(Stan3ArgsTest, GetInitFileForChain_SingleFile) {
  stan3::stan3_args args;
  args.init_files = {"shared.json"};
  
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "shared.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 1), "shared.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 10), "shared.json");
}

TEST(Stan3ArgsTest, GetInitFileForChain_MultipleFiles) {
  stan3::stan3_args args;
  args.init_files = {"init0.json", "init1.json", "init2.json"};
  
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "init0.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 1), "init1.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 2), "init2.json");
}

/* Test JSON file validators */
TEST(Stan3ArgsTest, JSONFileValidator_EmptyString) {
  stan3::JSONFileValidator validator;
  EXPECT_EQ(validator(""), "");
}

TEST(Stan3ArgsTest, JSONFileValidator_NonexistentFile) {
  stan3::JSONFileValidator validator;
  std::string result = validator("nonexistent.json");
  EXPECT_NE(result.find("does not exist"), std::string::npos);
}

TEST(Stan3ArgsTest, JSONFileValidator_ValidFile) {
  stan3::JSONFileValidator validator;
  EXPECT_EQ(validator("src/test/test-models/bernoulli.data.json"), "");
  EXPECT_EQ(validator("src/test/unit/json/valid_data.json"), "");
}

TEST(Stan3ArgsTest, JSONFileValidator_InvalidJSON) {
  stan3::JSONFileValidator validator;
  std::string result = validator("src/test/test-models/bernoulli.stan");
  EXPECT_NE(result.find("JSON object"), std::string::npos);
}

TEST(Stan3ArgsTest, JSONFileValidator_EmptyFile) {
  stan3::JSONFileValidator validator;
  std::string result = validator("src/test/unit/json/empty_data.json");
  EXPECT_NE(result.find("JSON object"), std::string::npos);
}

TEST(Stan3ArgsTest, JSONFileVectorValidator_EmptyString) {
  stan3::JSONFileVectorValidator validator;
  EXPECT_EQ(validator(""), "");
}

TEST(Stan3ArgsTest, JSONFileVectorValidator_ValidFile) {
  stan3::JSONFileVectorValidator validator;
  EXPECT_EQ(validator("src/test/test-models/bernoulli.data.json"), "");
}

TEST(Stan3ArgsTest, JSONFileVectorValidator_InvalidFile) {
  stan3::JSONFileVectorValidator validator;
  std::string result = validator("nonexistent.json");
  EXPECT_NE(result.find("does not exist"), std::string::npos);
}

/* Test string-to-enum mappings */
TEST(Stan3ArgsTest, CreateMetricMap) {
  auto map = stan3::create_metric_map();
  
  EXPECT_EQ(map["unit_e"], stan3::metric_t::UNIT_E);
  EXPECT_EQ(map["diag_e"], stan3::metric_t::DIAG_E);
  EXPECT_EQ(map["dense_e"], stan3::metric_t::DENSE_E);
}

/* Test temp directory functions */
TEST(Stan3ArgsTest, CreateTempOutputDir) {
  std::string temp_dir = stan3::create_temp_output_dir();
  
  EXPECT_FALSE(temp_dir.empty());
  EXPECT_NE(temp_dir.find("stan3_output_"), std::string::npos);
  
  // Clean up
  stan3::cleanup_temp_dir(temp_dir);
}

TEST(Stan3ArgsTest, CleanupTempDir_ValidDir) {
  std::string temp_dir = stan3::create_temp_output_dir();
  
  // Verify directory exists
  EXPECT_TRUE(std::filesystem::exists(temp_dir));
  
  // Clean up and verify it's gone
  stan3::cleanup_temp_dir(temp_dir);
  EXPECT_FALSE(std::filesystem::exists(temp_dir));
}

TEST(Stan3ArgsTest, CleanupTempDir_NonStanDir) {
  // Should not delete directories that don't match pattern
  std::string other_dir = "/tmp/some_other_dir";
  
  // This should not crash or delete anything
  EXPECT_NO_THROW(stan3::cleanup_temp_dir(other_dir));
}

TEST(Stan3ArgsTest, FinalizeArguments_EmptyOutputDir) {
  stan3::stan3_args args;
  args.output_dir = "";
  
  stan3::finalize_arguments(args);
  
  EXPECT_FALSE(args.output_dir.empty());
  EXPECT_NE(args.output_dir.find("stan3_output_"), std::string::npos);
  
  // Clean up
  stan3::cleanup_temp_dir(args.output_dir);
}

TEST(Stan3ArgsTest, FinalizeArguments_ExistingOutputDir) {
  stan3::stan3_args args;
  args.output_dir = "/existing/path";
  
  stan3::finalize_arguments(args);
  
  EXPECT_EQ(args.output_dir, "/existing/path");
}

/* Test stan3_args struct defaults */
TEST(Stan3ArgsTest, DefaultValues) {
  stan3::stan3_args args;
  
  EXPECT_EQ(args.algorithm, stan3::algorithm_t::STAN2_HMC);
  EXPECT_EQ(args.num_chains, 1);
  EXPECT_EQ(args.random_seed, 1);
  EXPECT_EQ(args.init_radius, 2.0);
  EXPECT_TRUE(args.data_file.empty());
  EXPECT_TRUE(args.init_files.empty());
  EXPECT_TRUE(args.output_dir.empty());
}

/* Test inheritance - hmc_nuts_args should inherit from stan3_args */
TEST(Stan3ArgsTest, HmcArgsInheritance) {
  stan3::hmc_nuts_args hmc_args;
  
  // Test that base class members are accessible
  hmc_args.num_chains = 4;
  hmc_args.random_seed = 12345;
  hmc_args.init_radius = 1.5;
  hmc_args.data_file = "test.json";
  
  EXPECT_EQ(hmc_args.num_chains, 4);
  EXPECT_EQ(hmc_args.random_seed, 12345);
  EXPECT_EQ(hmc_args.init_radius, 1.5);
  EXPECT_EQ(hmc_args.data_file, "test.json");
  
  // Test that HMC-specific members are also accessible
  hmc_args.num_warmup = 500;
  hmc_args.stepsize = 0.5;
  
  EXPECT_EQ(hmc_args.num_warmup, 500);
  EXPECT_EQ(hmc_args.stepsize, 0.5);
}

/* Test polymorphic usage */
TEST(Stan3ArgsTest, PolymorphicUsage) {
  stan3::hmc_nuts_args hmc_args;
  hmc_args.init_files = {"file1.json", "file2.json"};
  
  // Should be able to pass hmc_args to functions expecting stan3_args
  stan3::stan3_args& base_ref = hmc_args;
  
  EXPECT_EQ(stan3::get_init_file_for_chain(base_ref, 0), "file1.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(base_ref, 1), "file2.json");
}

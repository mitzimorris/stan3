#include <stan3/hmc_nuts_arguments.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

/* Test argument validation function */
TEST(HmcNutsArgsTest, ValidateArguments_ValidArgs) {
  stan3::hmc_nuts_args args;
  args.num_chains = 2;
  args.thin = 1;
  args.num_samples = 1000;
  
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_arguments(args, error_msg));
  EXPECT_TRUE(error_msg.empty());
}

TEST(HmcNutsArgsTest, ValidateArguments_ThinExceedsSamples) {
  stan3::hmc_nuts_args args;
  args.thin = 1500;
  args.num_samples = 1000;
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("thin"), std::string::npos);
  EXPECT_NE(error_msg.find("exceed"), std::string::npos);
}

TEST(HmcNutsArgsTest, ValidateArguments_InitFiles_WrongCount) {
  stan3::hmc_nuts_args args;
  args.num_chains = 3;
  args.init_files = {"file1.json", "file2.json"};  // Wrong count
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("inits"), std::string::npos);
}

TEST(HmcNutsArgsTest, ValidateArguments_InitFiles_ValidCounts) {
  stan3::hmc_nuts_args args;
  args.num_chains = 3;
  
  // Test single file for all chains
  args.init_files = {"file1.json"};
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_arguments(args, error_msg));
  
  // Test one file per chain
  args.init_files = {"file1.json", "file2.json", "file3.json"};
  EXPECT_TRUE(stan3::validate_arguments(args, error_msg));
  
  // Test empty (default initialization)
  args.init_files.clear();
  EXPECT_TRUE(stan3::validate_arguments(args, error_msg));
}

TEST(HmcNutsArgsTest, ValidateArguments_MetricFiles_WrongCount) {
  stan3::hmc_nuts_args args;
  args.num_chains = 2;
  args.metric_files = {"m1.json", "m2.json", "m3.json"};  // Wrong count
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("metric"), std::string::npos);
}

/* Test file path helper functions */
TEST(HmcNutsArgsTest, GetInitFileForChain_EmptyFiles) {
  stan3::hmc_nuts_args args;
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 5), "");
}

TEST(HmcNutsArgsTest, GetInitFileForChain_SingleFile) {
  stan3::hmc_nuts_args args;
  args.init_files = {"shared.json"};
  
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "shared.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 1), "shared.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 10), "shared.json");
}

TEST(HmcNutsArgsTest, GetInitFileForChain_MultipleFiles) {
  stan3::hmc_nuts_args args;
  args.init_files = {"init0.json", "init1.json", "init2.json"};
  
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 0), "init0.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 1), "init1.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(args, 2), "init2.json");
}

TEST(HmcNutsArgsTest, GetMetricFileForChain_EmptyFiles) {
  stan3::hmc_nuts_args args;
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 0), "");
}

TEST(HmcNutsArgsTest, GetMetricFileForChain_SingleFile) {
  stan3::hmc_nuts_args args;
  args.metric_files = {"shared_metric.json"};
  
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 0), "shared_metric.json");
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 3), "shared_metric.json");
}

/* Test JSON file validators using existing test files */
TEST(HmcNutsArgsTest, JSONFileValidator_EmptyString) {
  stan3::JSONFileValidator validator;
  EXPECT_EQ(validator(""), "");
}

TEST(HmcNutsArgsTest, JSONFileValidator_NonexistentFile) {
  stan3::JSONFileValidator validator;
  std::string result = validator("nonexistent.json");
  EXPECT_NE(result.find("does not exist"), std::string::npos);
}

TEST(HmcNutsArgsTest, JSONFileValidator_ValidFile) {
  stan3::JSONFileValidator validator;
  EXPECT_EQ(validator("src/test/test-models/bernoulli.data.json"), "");
  EXPECT_EQ(validator("src/test/unit/json/valid_data.json"), "");
}

TEST(HmcNutsArgsTest, JSONFileValidator_InvalidJSON) {
  stan3::JSONFileValidator validator;
  std::string result = validator("src/test/test-models/bernoulli.stan");
  EXPECT_NE(result.find("JSON object"), std::string::npos);
}

TEST(HmcNutsArgsTest, JSONFileValidator_EmptyFile) {
  stan3::JSONFileValidator validator;
  std::string result = validator("src/test/unit/json/empty_data.json");
  EXPECT_NE(result.find("JSON object"), std::string::npos);
}

/* Test string-to-enum mappings */
TEST(HmcNutsArgsTest, CreateAlgorithmMap) {
  auto map = stan3::create_algorithm_map();
  
  EXPECT_EQ(map["hmc"], stan3::algorithm_t::STAN2_HMC);
  EXPECT_EQ(map["mle"], stan3::algorithm_t::MLE);
  EXPECT_EQ(map["pathfinder"], stan3::algorithm_t::PATHFINDER);
  EXPECT_EQ(map["advi"], stan3::algorithm_t::ADVI);
  EXPECT_EQ(map["gq"], stan3::algorithm_t::STANDALONE_GQ);
}

TEST(HmcNutsArgsTest, CreateMetricMap) {
  auto map = stan3::create_metric_map();
  
  EXPECT_EQ(map["unit_e"], stan3::metric_t::UNIT_E);
  EXPECT_EQ(map["diag_e"], stan3::metric_t::DIAG_E);
  EXPECT_EQ(map["dense_e"], stan3::metric_t::DENSE_E);
}

/* Test temp directory functions */
TEST(HmcNutsArgsTest, CreateTempOutputDir) {
  std::string temp_dir = stan3::create_temp_output_dir();
  
  EXPECT_FALSE(temp_dir.empty());
  EXPECT_NE(temp_dir.find("stan3_output_"), std::string::npos);
  
  // Clean up
  stan3::cleanup_temp_dir(temp_dir);
}

TEST(HmcNutsArgsTest, FinalizeArguments_EmptyOutputDir) {
  stan3::hmc_nuts_args args;
  args.output_dir = "";
  
  stan3::finalize_arguments(args);
  
  EXPECT_FALSE(args.output_dir.empty());
  EXPECT_NE(args.output_dir.find("stan3_output_"), std::string::npos);
  
  // Clean up
  stan3::cleanup_temp_dir(args.output_dir);
}

TEST(HmcNutsArgsTest, FinalizeArguments_ExistingOutputDir) {
  stan3::hmc_nuts_args args;
  args.output_dir = "/existing/path";
  
  stan3::finalize_arguments(args);
  
  EXPECT_EQ(args.output_dir, "/existing/path");
}

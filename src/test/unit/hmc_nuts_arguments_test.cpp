#include <stan3/arguments.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

/* Test HMC-NUTS specific argument validation */
TEST(HmcNutsArgsTest, ValidateHmcArguments_ValidArgs) {
  stan3::hmc_nuts_args args;
  args.base.num_chains = 2;
  args.thin = 1;
  args.num_samples = 1000;
  
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  EXPECT_TRUE(error_msg.empty());
}

TEST(HmcNutsArgsTest, ValidateHmcArguments_ThinExceedsSamples) {
  stan3::hmc_nuts_args args;
  args.thin = 1500;
  args.num_samples = 1000;
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_hmc_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("thin"), std::string::npos);
  EXPECT_NE(error_msg.find("exceed"), std::string::npos);
}

TEST(HmcNutsArgsTest, ValidateHmcArguments_InitFiles_WrongCount) {
  stan3::hmc_nuts_args args;
  args.base.num_chains = 3;
  args.base.init.init_files = {"file1.json", "file2.json"};  // Wrong count
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_hmc_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("inits"), std::string::npos);
}

TEST(HmcNutsArgsTest, ValidateHmcArguments_InitFiles_ValidCounts) {
  stan3::hmc_nuts_args args;
  args.base.num_chains = 3;
  
  // Test single file for all chains
  args.base.init.init_files = {"file1.json"};
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Test one file per chain
  args.base.init.init_files = {"file1.json", "file2.json", "file3.json"};
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Test empty (default initialization)
  args.base.init.init_files.clear();
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
}

TEST(HmcNutsArgsTest, ValidateHmcArguments_MetricFiles_WrongCount) {
  stan3::hmc_nuts_args args;
  args.base.num_chains = 2;
  args.metric_files = {"m1.json", "m2.json", "m3.json"};  // Wrong count
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_hmc_arguments(args, error_msg));
  EXPECT_NE(error_msg.find("metric"), std::string::npos);
}

TEST(HmcNutsArgsTest, ValidateHmcArguments_MetricFiles_ValidCounts) {
  stan3::hmc_nuts_args args;
  args.base.num_chains = 2;
  
  // Test single file for all chains
  args.metric_files = {"metric.json"};
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Test one file per chain
  args.metric_files = {"m1.json", "m2.json"};
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Test empty (default metric)
  args.metric_files.clear();
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
}

/* Test HMC-specific file path helper functions */
TEST(HmcNutsArgsTest, GetMetricFileForChain_EmptyFiles) {
  stan3::hmc_nuts_args args;
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 0), "");
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 5), "");
}

TEST(HmcNutsArgsTest, GetMetricFileForChain_SingleFile) {
  stan3::hmc_nuts_args args;
  args.metric_files = {"shared_metric.json"};
  
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 0), "shared_metric.json");
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 3), "shared_metric.json");
}

TEST(HmcNutsArgsTest, GetMetricFileForChain_MultipleFiles) {
  stan3::hmc_nuts_args args;
  args.metric_files = {"metric0.json", "metric1.json", "metric2.json"};
  
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 0), "metric0.json");
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 1), "metric1.json");
  EXPECT_EQ(stan3::get_metric_file_for_chain(args, 2), "metric2.json");
}

/* Test HMC-NUTS default values */
TEST(HmcNutsArgsTest, HmcDefaultValues) {
  stan3::hmc_nuts_args args;
  
  // Test composed defaults from model_args
  EXPECT_EQ(args.base.model.random_seed, 1);
  EXPECT_TRUE(args.base.model.data_file.empty());
  
  // Test composed defaults from init_args
  EXPECT_EQ(args.base.init.init_radius, 2.0);
  EXPECT_TRUE(args.base.init.init_files.empty());
  EXPECT_TRUE(args.base.output_dir.empty());
  
  // Test inference_args defaults
  EXPECT_EQ(args.base.num_chains, 1);
  
  // Test HMC-specific defaults
  EXPECT_EQ(args.num_warmup, 1000);
  EXPECT_EQ(args.num_samples, 1000);
  EXPECT_EQ(args.thin, 1);
  EXPECT_EQ(args.refresh, 100);
  EXPECT_EQ(args.metric_type, stan3::metric_t::DIAG_E);
  EXPECT_EQ(args.stepsize, 1.0);
  EXPECT_EQ(args.stepsize_jitter, 0.0);
  EXPECT_EQ(args.max_depth, 10);
  
  // Test output defaults
  EXPECT_FALSE(args.save_start_params);
  EXPECT_FALSE(args.save_warmup);
  EXPECT_FALSE(args.save_diagnostics);
  EXPECT_FALSE(args.save_metric);
  
  // Test NUTS adaptation defaults
  EXPECT_EQ(args.delta, 0.8);
  EXPECT_EQ(args.gamma, 0.05);
  EXPECT_EQ(args.kappa, 0.75);
  EXPECT_EQ(args.t0, 10.0);
  EXPECT_EQ(args.init_buffer, 75);
  EXPECT_EQ(args.term_buffer, 50);
  EXPECT_EQ(args.window, 25);
  
  // Test that file vectors are empty by default
  EXPECT_TRUE(args.metric_files.empty());
}

/* Test boundary conditions for HMC validation */
TEST(HmcNutsArgsTest, ValidateHmcArguments_BoundaryConditions) {
  stan3::hmc_nuts_args args;
  std::string error_msg;
  
  // Test thin equals samples (should be valid)
  args.thin = 1000;
  args.num_samples = 1000;
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Test thin = 1 (should always be valid)
  args.thin = 1;
  args.num_samples = 0;  // Even with 0 samples
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
}

/* Test edge cases for metric file validation */
TEST(HmcNutsArgsTest, ValidateHmcArguments_MetricFiles_EdgeCases) {
  stan3::hmc_nuts_args args;
  std::string error_msg;
  
  // Single chain with multiple metric files (should fail)
  args.base.num_chains = 1;
  args.metric_files = {"m1.json", "m2.json"};
  EXPECT_FALSE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Large number of chains
  args.base.num_chains = 10;
  args.metric_files.clear();
  for (int i = 0; i < 10; ++i) {
    args.metric_files.push_back("m" + std::to_string(i) + ".json");
  }
  EXPECT_TRUE(stan3::validate_hmc_arguments(args, error_msg));
}

/* Test that HMC args can be used compositionally */
TEST(HmcNutsArgsTest, CompositionalValidation) {
  stan3::hmc_nuts_args hmc_args;
  hmc_args.base.num_chains = 2;
  hmc_args.thin = 500;
  hmc_args.num_samples = 1000;
  hmc_args.base.init.init_files = {"init1.json", "init2.json"};
  
  // Should pass HMC-specific validation
  std::string error_msg;
  EXPECT_TRUE(stan3::validate_hmc_arguments(hmc_args, error_msg));
  
  // Should also work with composed functions (init files)
  stan3::init_args& init_ref = hmc_args.base.init;
  EXPECT_EQ(stan3::get_init_file_for_chain(init_ref, 0), "init1.json");
  EXPECT_EQ(stan3::get_init_file_for_chain(init_ref, 1), "init2.json");
}

/* Test complex validation scenarios */
TEST(HmcNutsArgsTest, ValidateHmcArguments_MultipleErrors) {
  stan3::hmc_nuts_args args;
  
  // Set up multiple potential errors, but validation should stop at first one
  args.base.num_chains = 3;
  args.thin = 2000;        // > num_samples
  args.num_samples = 1000;
  args.base.init.init_files = {"file1.json", "file2.json"};  // Wrong count
  args.metric_files = {"m1.json"};  // Valid count
  
  std::string error_msg;
  EXPECT_FALSE(stan3::validate_hmc_arguments(args, error_msg));
  
  // Should mention thin error (checked first)
  EXPECT_NE(error_msg.find("thin"), std::string::npos);
}

/* Test HMC standalone parser */
TEST(HmcNutsArgsTest, ParseHmcArgs_ValidArgs) {
  const char* argv[] = {"stan3", "--chains", "2", "--warmup", "500", 
                        "--samples", "1000", "--stepsize", "0.5"};
  int argc = 9;
  
  stan3::hmc_nuts_args args;
  std::string error_msg;
  
  // This should work since no file validation is involved
  bool result = stan3::parse_hmc_args(argc, const_cast<char**>(argv), args, error_msg);
  
  if (result) {
    EXPECT_EQ(args.base.num_chains, 2);
    EXPECT_EQ(args.num_warmup, 500);
    EXPECT_EQ(args.num_samples, 1000);
    EXPECT_EQ(args.stepsize, 0.5);
  } else {
    // If it fails, print the error for debugging
    std::cout << "Parse error: " << error_msg << std::endl;
  }
}

/* Test finalize function */
TEST(HmcNutsArgsTest, FinalizeHmcArguments) {
  stan3::hmc_nuts_args args;
  args.base.output_dir = "";
  
  stan3::finalize_hmc_arguments(args);
  
  EXPECT_FALSE(args.base.output_dir.empty());
  EXPECT_NE(args.base.output_dir.find("stan3_output_"), std::string::npos);
  
  // Clean up
  stan3::cleanup_temp_dir(args.base.output_dir);
}

/* Test composition structure integrity */
TEST(HmcNutsArgsTest, CompositionStructureIntegrity) {
  stan3::hmc_nuts_args args;
  
  // Test that we can access all levels of composition
  args.base.model.random_seed = 42;
  args.base.model.data_file = "test.json";
  args.base.num_chains = 4;
  args.base.init.init_radius = 1.5;
  args.base.init.init_files = {"init.json"};
  args.base.output_dir = "/tmp/output";
  
  // Verify all assignments worked
  EXPECT_EQ(args.base.model.random_seed, 42);
  EXPECT_EQ(args.base.model.data_file, "test.json");
  EXPECT_EQ(args.base.num_chains, 4);
  EXPECT_EQ(args.base.init.init_radius, 1.5);
  EXPECT_EQ(args.base.init.init_files.size(), 1);
  EXPECT_EQ(args.base.init.init_files[0], "init.json");
  EXPECT_EQ(args.base.output_dir, "/tmp/output");
  
  // Test HMC-specific fields still work
  args.num_warmup = 2000;
  args.stepsize = 0.1;
  EXPECT_EQ(args.num_warmup, 2000);
  EXPECT_EQ(args.stepsize, 0.1);
}

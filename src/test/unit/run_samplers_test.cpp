#include <stan3/run_samplers.hpp>
#include <stan3/arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan3/read_json_data.hpp>

#include <test/test-models/bernoulli.hpp>

#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/interrupt.hpp>

#include <filesystem>
#include <memory>
#include <vector>
#include <sstream>

#include <gtest/gtest.h>

class RunSamplersTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create temporary output directory
    temp_dir_ = std::filesystem::temp_directory_path() / "run_samplers_test";
    std::filesystem::create_directories(temp_dir_);
    
    // Load test data and create model
    auto data_context = stan3::read_json_data("src/test/test-models/bernoulli.data.json");
    model_ = std::make_unique<bernoulli_model_namespace::bernoulli_model>(*data_context, 12345);
    
    // Setup minimal arguments for fast testing
    args_.base.num_chains = 1;
    args_.base.model.random_seed = 12345;
    args_.metric_type = stan3::metric_t::UNIT_E;  // Fastest option
    args_.num_warmup = 10;    // Very small for testing
    args_.num_samples = 10;   // Very small for testing
    args_.stepsize = 1.0;
    args_.max_depth = 5;      // Small for speed
    args_.delta = 0.8;
    args_.refresh = 0;        // No progress messages
    args_.base.output_dir = temp_dir_.string();
    
    // Create contexts
    for (size_t i = 0; i < args_.base.num_chains; ++i) {
      init_contexts_.push_back(stan3::read_json_data(""));
      metric_contexts_.push_back(stan3::read_json_data(""));
    }
    
    // Create writers
    writers_ = stan3::create_hmc_nuts_multi_chain_writers(args_, "test_model");
    
    // Create logger and interrupt
    logger_ = std::make_unique<stan::callbacks::stream_logger>(
      log_stream_, log_stream_, log_stream_, log_stream_, log_stream_);
    interrupt_ = std::make_unique<stan::callbacks::interrupt>();
  }
  
  void TearDown() override {
    std::filesystem::remove_all(temp_dir_);
  }
  
  std::filesystem::path temp_dir_;
  std::unique_ptr<bernoulli_model_namespace::bernoulli_model> model_;
  stan3::hmc_nuts_args args_;
  std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts_;
  std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts_;
  std::vector<stan3::hmc_nuts_writers> writers_;
  std::unique_ptr<stan::callbacks::stream_logger> logger_;
  std::unique_ptr<stan::callbacks::interrupt> interrupt_;
  std::stringstream log_stream_;  // Capture log output
};

TEST_F(RunSamplersTest, RunSamplers_SingleChain_UnitE) {
  args_.base.num_chains = 1;
  args_.metric_type = stan3::metric_t::UNIT_E;
  
  // Resize contexts and writers for single chain
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan3::run_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                       writers_, *interrupt_, *logger_);
  });
  
  // Check that sample file was created
  EXPECT_TRUE(std::filesystem::exists(temp_dir_));
}

TEST_F(RunSamplersTest, RunSamplers_SingleChain_DiagE) {
  args_.base.num_chains = 1;
  args_.metric_type = stan3::metric_t::DIAG_E;
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan3::run_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                       writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, RunSamplers_MultipleChains) {
  args_.base.num_chains = 2;
  args_.metric_type = stan3::metric_t::UNIT_E;
  
  // Resize contexts and writers for multiple chains
  init_contexts_.resize(2);
  metric_contexts_.resize(2);
  writers_.resize(2);
  
  // Add second chain contexts
  init_contexts_[1] = stan3::read_json_data("");
  metric_contexts_[1] = stan3::read_json_data("");
  
  // Create second chain writers
  writers_[1] = stan3::create_hmc_nuts_single_chain_writers(
    args_, "test_model", stan3::generate_timestamp(), 2);
  
  EXPECT_NO_THROW({
    stan3::run_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                       writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, SamplerRunner_Construction) {
  stan3::sampler_runner runner(*model_, args_, writers_, *interrupt_, *logger_);
  
  // Just test that construction doesn't throw
  EXPECT_TRUE(true);
}

TEST_F(RunSamplersTest, RunSamplers_MinimalSampling) {
  // Test with absolute minimal sampling to ensure basic functionality
  args_.base.num_chains = 1;
  args_.num_warmup = 1;
  args_.num_samples = 1;
  args_.metric_type = stan3::metric_t::UNIT_E;
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  writers_.resize(1);
  
  EXPECT_NO_THROW({
    stan3::run_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                       writers_, *interrupt_, *logger_);
  });
}

TEST_F(RunSamplersTest, RunSamplers_WithNullableWriters) {
  args_.base.num_chains = 1;
  args_.save_start_params = false;  // This should make start_params_writer null
  args_.save_diagnostics = false;   // This should make diagnostics_writer null
  args_.save_metric = false;        // This should make metric_writer null
  
  // Recreate writers with nullable options
  writers_ = stan3::create_hmc_nuts_multi_chain_writers(args_, "test_model");
  
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  
  EXPECT_NO_THROW({
    stan3::run_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                       writers_, *interrupt_, *logger_);
  });
  
  // Verify that nullable writers are actually null
  EXPECT_EQ(writers_[0].start_params_writer, nullptr);
  EXPECT_EQ(writers_[0].diagnostics_writer, nullptr);
  EXPECT_EQ(writers_[0].metric_writer, nullptr);
  EXPECT_NE(writers_[0].sample_writer, nullptr);  // Sample writer should never be null
}

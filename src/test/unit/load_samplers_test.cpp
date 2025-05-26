#include <stan3/load_samplers.hpp>
#include <stan3/arguments.hpp>
#include <stan3/read_json_data.hpp>

#include <test/test-models/bernoulli.hpp>

#include <stan/callbacks/stream_logger.hpp>

#include <memory>
#include <vector>

#include <gtest/gtest.h>

class LoadSamplersTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Load test data and create model
    auto data_context = stan3::read_json_data("src/test/test-models/bernoulli.data.json");
    model_ = std::make_unique<bernoulli_model_namespace::bernoulli_model>(*data_context, 12345);
    
    // Setup basic arguments
    args_.base.num_chains = 2;
    args_.base.model.random_seed = 12345;
    args_.metric_type = stan3::metric_t::DIAG_E;
    args_.stepsize = 1.0;
    args_.max_depth = 10;
    args_.delta = 0.8;
    
    // Create empty init contexts (use default initialization)
    for (size_t i = 0; i < args_.base.num_chains; ++i) {
      init_contexts_.push_back(stan3::read_json_data(""));
      metric_contexts_.push_back(stan3::read_json_data(""));
      init_writers_.push_back(nullptr);
    }
    
    logger_ = std::make_unique<stan::callbacks::stream_logger>(
      std::cout, std::cout, std::cout, std::cerr, std::cerr);
  }
  
  std::unique_ptr<bernoulli_model_namespace::bernoulli_model> model_;
  stan3::hmc_nuts_args args_;
  std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts_;
  std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts_;
  std::vector<stan::callbacks::writer*> init_writers_;
  std::unique_ptr<stan::callbacks::stream_logger> logger_;
};

TEST_F(LoadSamplersTest, CreateSamplers_DiagE) {
  args_.metric_type = stan3::metric_t::DIAG_E;
  
  auto sampler_configs = stan3::create_samplers(*model_, args_, init_contexts_, 
                                               metric_contexts_, *logger_, init_writers_);
  
  // Should return a variant - we can't easily check the exact type, 
  // but we can verify it doesn't throw
  EXPECT_NO_THROW({
    std::visit([this](auto& config) {
      EXPECT_EQ(config.samplers.size(), args_.base.num_chains);
      EXPECT_EQ(config.rngs.size(), args_.base.num_chains);
      EXPECT_EQ(config.init_params.size(), args_.base.num_chains);
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, CreateSamplers_UnitE) {
  args_.metric_type = stan3::metric_t::UNIT_E;
  
  EXPECT_NO_THROW({
    auto sampler_configs = stan3::create_samplers(*model_, args_, init_contexts_, 
                                                 metric_contexts_, *logger_, init_writers_);
    
    std::visit([this](auto& config) {
      EXPECT_EQ(config.samplers.size(), args_.base.num_chains);
      EXPECT_EQ(config.rngs.size(), args_.base.num_chains);
      EXPECT_EQ(config.init_params.size(), args_.base.num_chains);
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, CreateSamplers_DenseE) {
  args_.metric_type = stan3::metric_t::DENSE_E;
  
  EXPECT_NO_THROW({
    auto sampler_configs = stan3::create_samplers(*model_, args_, init_contexts_, 
                                                 metric_contexts_, *logger_, init_writers_);
    
    std::visit([this](auto& config) {
      EXPECT_EQ(config.samplers.size(), args_.base.num_chains);
      EXPECT_EQ(config.rngs.size(), args_.base.num_chains);
      EXPECT_EQ(config.init_params.size(), args_.base.num_chains);
    }, sampler_configs);
  });
}

TEST_F(LoadSamplersTest, CreateSamplers_InvalidMetricType) {
  args_.metric_type = static_cast<stan3::metric_t>(999);  // Invalid metric type
  
  EXPECT_THROW({
    stan3::create_samplers(*model_, args_, init_contexts_, metric_contexts_, 
                          *logger_, init_writers_);
  }, std::runtime_error);
}

TEST_F(LoadSamplersTest, LoadSamplers_DiagE_CorrectSizes) {
  auto config = stan3::load_samplers<stan3::metric_t::DIAG_E>(
    *model_, args_, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(config.samplers.size(), args_.base.num_chains);
  EXPECT_EQ(config.rngs.size(), args_.base.num_chains);
  EXPECT_EQ(config.init_params.size(), args_.base.num_chains);
  
  // Check that init_params have the right size for each chain
  for (size_t i = 0; i < args_.base.num_chains; ++i) {
    EXPECT_EQ(config.init_params[i].size(), model_->num_params_r());
  }
}

TEST_F(LoadSamplersTest, LoadSamplers_UnitE_CorrectSizes) {
  auto config = stan3::load_samplers<stan3::metric_t::UNIT_E>(
    *model_, args_, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(config.samplers.size(), args_.base.num_chains);
  EXPECT_EQ(config.rngs.size(), args_.base.num_chains);
  EXPECT_EQ(config.init_params.size(), args_.base.num_chains);
  
  for (size_t i = 0; i < args_.base.num_chains; ++i) {
    EXPECT_EQ(config.init_params[i].size(), model_->num_params_r());
  }
}

TEST_F(LoadSamplersTest, LoadSamplers_SingleChain) {
  args_.base.num_chains = 1;
  
  // Resize contexts for single chain
  init_contexts_.resize(1);
  metric_contexts_.resize(1);
  init_writers_.resize(1);
  
  auto config = stan3::load_samplers<stan3::metric_t::DIAG_E>(
    *model_, args_, init_contexts_, metric_contexts_, *logger_, init_writers_);
  
  EXPECT_EQ(config.samplers.size(), 1);
  EXPECT_EQ(config.rngs.size(), 1);
  EXPECT_EQ(config.init_params.size(), 1);
  EXPECT_EQ(config.init_params[0].size(), model_->num_params_r());
}

TEST_F(LoadSamplersTest, SamplerTraits_TypeAliases) {
  // Test that sampler_traits compile correctly
  using diag_sampler = stan3::sampler_traits<stan3::metric_t::DIAG_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  using unit_sampler = stan3::sampler_traits<stan3::metric_t::UNIT_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  using dense_sampler = stan3::sampler_traits<stan3::metric_t::DENSE_E>::sampler_type<bernoulli_model_namespace::bernoulli_model>;
  
  // Just check that these types compile - we can't easily instantiate them here
  EXPECT_TRUE(true);  // Placeholder assertion
}

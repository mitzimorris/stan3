#include <stan3/load_samplers.hpp>
#include <stan3/load_model.hpp>
#include <stan3/read_json_data.hpp>
#include <stan3/hmc_nuts_arguments.hpp>
#include <stan3/hmc_output_writers.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>

#include <test/test-models/bernoulli.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

class LoadSamplersTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir = std::filesystem::temp_directory_path() / "stan3_load_samplers_test";
    std::filesystem::create_directories(test_dir);
    
    init_file = test_dir / "init.json";
    std::ofstream init_stream(init_file);
    init_stream << R"({ "theta": 0.5 })";
    init_stream.close();
    
    diag_metric_file = test_dir / "diag_metric.json";
    std::ofstream diag_stream(diag_metric_file);
    diag_stream << R"({ "inv_metric": [1.5] })";
    diag_stream.close();
    
    dense_metric_file = test_dir / "dense_metric.json";
    std::ofstream dense_stream(dense_metric_file);
    dense_stream << R"({ "inv_metric": [[2.0]] })";
    dense_stream.close();
    
    logger = std::make_unique<stan::callbacks::stream_logger>(
      std::cout, std::cout, std::cout, std::cerr, std::cerr);
    
    // Load model and data
    data_context = stan3::read_json_data("src/test/test-models/bernoulli.data.json");
    stan::io::var_context* raw_context = const_cast<stan::io::var_context*>(data_context.get());
    model = stan3::load_model(*raw_context, 12345);
  }
  
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
  }
  
  /* Helper to create init contexts */
  std::vector<std::shared_ptr<const stan::io::var_context>>
  create_init_contexts(size_t num_chains, bool use_file = true) {
    std::vector<std::shared_ptr<const stan::io::var_context>> contexts;
    contexts.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      if (use_file) {
        contexts.push_back(stan3::read_json_data(init_file.string()));
      } else {
        contexts.push_back(std::make_shared<stan::io::empty_var_context>());
      }
    }
    return contexts;
  }
  
  /* Helper to create metric contexts */
  std::vector<std::shared_ptr<const stan::io::var_context>>
  create_metric_contexts(size_t num_chains, stan3::metric_t metric_type) {
    std::vector<std::shared_ptr<const stan::io::var_context>> contexts;
    contexts.reserve(num_chains);
    
    for (size_t i = 0; i < num_chains; ++i) {
      switch (metric_type) {
        case stan3::metric_t::UNIT_E:
          contexts.push_back(std::make_shared<stan::io::empty_var_context>());
          break;
        case stan3::metric_t::DIAG_E:
          contexts.push_back(stan3::read_json_data(diag_metric_file.string()));
          break;
        case stan3::metric_t::DENSE_E:
          contexts.push_back(stan3::read_json_data(dense_metric_file.string()));
          break;
      }
    }
    return contexts;
  }
  
  /* Helper to create init writers */
  std::vector<stan::callbacks::writer*> create_init_writers(size_t num_chains) {
    dummy_writers.clear();
    dummy_writers.resize(num_chains);
    
    std::vector<stan::callbacks::writer*> writers;
    writers.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      writers.push_back(&dummy_writers[i]);
    }
    return writers;
  }
  
  std::filesystem::path test_dir;
  std::filesystem::path init_file;
  std::filesystem::path diag_metric_file;
  std::filesystem::path dense_metric_file;
  
  std::unique_ptr<stan::callbacks::logger> logger;
  std::shared_ptr<const stan::io::var_context> data_context;
  std::unique_ptr<stan::model::model_base> model;
  stan3::hmc_nuts_args args;
  
  // Storage for dummy writers
  std::vector<stan::callbacks::writer> dummy_writers;
};

TEST_F(LoadSamplersTest, LoadSamplersUnitE) {
  args.metric_type = stan3::metric_t::UNIT_E;
  args.num_chains = 1;
  
  auto init_contexts = create_init_contexts(args.num_chains);
  auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
  auto init_writers = create_init_writers(args.num_chains);
  
  EXPECT_NO_THROW({
    auto config = stan3::load_samplers<stan3::metric_t::UNIT_E>(
      *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
    EXPECT_EQ(config.samplers.size(), 1);
    EXPECT_EQ(config.rngs.size(), 1);
    EXPECT_EQ(config.init_params.size(), 1);
    EXPECT_FALSE(config.init_params[0].empty());
  });
}

TEST_F(LoadSamplersTest, LoadSamplersDiagE) {
  args.metric_type = stan3::metric_t::DIAG_E;
  args.num_chains = 1;
  
  auto init_contexts = create_init_contexts(args.num_chains);
  auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
  auto init_writers = create_init_writers(args.num_chains);
  
  EXPECT_NO_THROW({
    auto config = stan3::load_samplers<stan3::metric_t::DIAG_E>(
      *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
    EXPECT_EQ(config.samplers.size(), 1);
    EXPECT_EQ(config.rngs.size(), 1);
    EXPECT_EQ(config.init_params.size(), 1);
    EXPECT_FALSE(config.init_params[0].empty());
  });
}

TEST_F(LoadSamplersTest, LoadSamplersDenseE) {
  args.metric_type = stan3::metric_t::DENSE_E;
  args.num_chains = 1;
  
  auto init_contexts = create_init_contexts(args.num_chains);
  auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
  auto init_writers = create_init_writers(args.num_chains);
  
  EXPECT_NO_THROW({
    auto config = stan3::load_samplers<stan3::metric_t::DENSE_E>(
      *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
    EXPECT_EQ(config.samplers.size(), 1);
    EXPECT_EQ(config.rngs.size(), 1);
    EXPECT_EQ(config.init_params.size(), 1);
    EXPECT_FALSE(config.init_params[0].empty());
  });
}

TEST_F(LoadSamplersTest, LoadSamplersMultipleChains) {
  args.metric_type = stan3::metric_t::DIAG_E;
  args.num_chains = 3;
  
  auto init_contexts = create_init_contexts(args.num_chains);
  auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
  auto init_writers = create_init_writers(args.num_chains);
  
  EXPECT_NO_THROW({
    auto config = stan3::load_samplers<stan3::metric_t::DIAG_E>(
      *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
    EXPECT_EQ(config.samplers.size(), 3);
    EXPECT_EQ(config.rngs.size(), 3);
    EXPECT_EQ(config.init_params.size(), 3);
    
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_FALSE(config.init_params[i].empty());
    }
  });
}

// TEST_F(LoadSamplersTest, CreateSamplersVariantUnitE) {
//   args.metric_type = stan3::metric_t::UNIT_E;
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_NO_THROW({
//     auto variant = stan3::create_samplers(*model, args, init_contexts, 
//                                         metric_contexts, *logger, init_writers);
    
//     // Should hold unit_e config
//     EXPECT_TRUE(std::holds_alternative<
//       stan3::sampler_config<stan::mcmc::adapt_unit_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant));
    
//     auto& config = std::get<
//       stan3::sampler_config<stan::mcmc::adapt_unit_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant);
    
//     EXPECT_EQ(config.samplers.size(), 1);
//   });
// }

// TEST_F(LoadSamplersTest, CreateSamplersVariantDiagE) {
//   args.metric_type = stan3::metric_t::DIAG_E;
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_NO_THROW({
//     auto variant = stan3::create_samplers(*model, args, init_contexts, 
//                                         metric_contexts, *logger, init_writers);
    
//     // Should hold diag_e config
//     EXPECT_TRUE(std::holds_alternative<
//       stan3::sampler_config<stan::mcmc::adapt_diag_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant));
    
//     auto& config = std::get<
//       stan3::sampler_config<stan::mcmc::adapt_diag_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant);
    
//     EXPECT_EQ(config.samplers.size(), 1);
//   });
// }

// TEST_F(LoadSamplersTest, CreateSamplersVariantDenseE) {
//   args.metric_type = stan3::metric_t::DENSE_E;
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_NO_THROW({
//     auto variant = stan3::create_samplers(*model, args, init_contexts, 
//                                         metric_contexts, *logger, init_writers);
    
//     // Should hold dense_e config
//     EXPECT_TRUE(std::holds_alternative<
//       stan3::sampler_config<stan::mcmc::adapt_dense_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant));
    
//     auto& config = std::get<
//       stan3::sampler_config<stan::mcmc::adapt_dense_e_nuts<stan::model::model_base, stan3::rng_t>>
//     >(variant);
    
//     EXPECT_EQ(config.samplers.size(), 1);
//   });
// }

// TEST_F(LoadSamplersTest, LoadSamplersWithEmptyInitContext) {
//   args.metric_type = stan3::metric_t::UNIT_E;
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains, false);  // use empty context
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_NO_THROW({
//     auto config = stan3::load_samplers<stan3::metric_t::UNIT_E>(
//       *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
//     EXPECT_EQ(config.samplers.size(), 1);
//     EXPECT_FALSE(config.init_params[0].empty());
//   });
// }

// TEST_F(LoadSamplersTest, LoadSamplersWithNullInitWriter) {
//   args.metric_type = stan3::metric_t::UNIT_E;
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
  
//   // Create init writers with null pointer
//   std::vector<stan::callbacks::writer*> init_writers = {nullptr};
  
//   EXPECT_NO_THROW({
//     auto config = stan3::load_samplers<stan3::metric_t::UNIT_E>(
//       *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
//     EXPECT_EQ(config.samplers.size(), 1);
//     EXPECT_FALSE(config.init_params[0].empty());
//   });
// }

// TEST_F(LoadSamplersTest, SamplerConfigurationParameters) {
//   args.metric_type = stan3::metric_t::DIAG_E;
//   args.num_chains = 1;
//   args.stepsize = 1.5;
//   args.stepsize_jitter = 0.1;
//   args.max_depth = 12;
//   args.delta = 0.85;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_NO_THROW({
//     auto config = stan3::load_samplers<stan3::metric_t::DIAG_E>(
//       *model, args, init_contexts, metric_contexts, *logger, init_writers);
    
//     EXPECT_EQ(config.samplers.size(), 1);
    
//     // Verify sampler is configured with correct parameters
//     auto& sampler = config.samplers[0];
    
//     // These are harder to test directly, but we can at least verify
//     // the sampler was created and configured without throwing
//     EXPECT_NO_THROW({
//       // Try to access adaptation parameters
//       auto& adaptation = sampler.get_stepsize_adaptation();
//       // If we get here, the sampler was properly configured
//     });
//   });
// }

// TEST_F(LoadSamplersTest, ErrorHandlingInvalidMetricType) {
//   // Test with an invalid metric type cast
//   args.metric_type = static_cast<stan3::metric_t>(999);
//   args.num_chains = 1;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, stan3::metric_t::UNIT_E);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   EXPECT_THROW({
//     auto variant = stan3::create_samplers(*model, args, init_contexts, 
//                                         metric_contexts, *logger, init_writers);
//   }, std::runtime_error);
// }

// TEST_F(LoadSamplersTest, ConsistentRNGSeeds) {
//   args.metric_type = stan3::metric_t::UNIT_E;
//   args.num_chains = 3;
//   args.random_seed = 42;
  
//   auto init_contexts = create_init_contexts(args.num_chains);
//   auto metric_contexts = create_metric_contexts(args.num_chains, args.metric_type);
//   auto init_writers = create_init_writers(args.num_chains);
  
//   // Create samplers twice with same seed
//   auto config1 = stan3::load_samplers<stan3::metric_t::UNIT_E>(
//     *model, args, init_contexts, metric_contexts, *logger, init_writers);
  
//   auto config2 = stan3::load_samplers<stan3::metric_t::UNIT_E>(
//     *model, args, init_contexts, metric_contexts, *logger, init_writers);
  
//   EXPECT_EQ(config1.samplers.size(), config2.samplers.size());
//   EXPECT_EQ(config1.rngs.size(), config2.rngs.size());
  
//   // Initial parameters should be the same with same seed and init context
//   for (size_t i = 0; i < config1.init_params.size(); ++i) {
//     EXPECT_EQ(config1.init_params[i].size(), config2.init_params[i].size());
//   }
// }


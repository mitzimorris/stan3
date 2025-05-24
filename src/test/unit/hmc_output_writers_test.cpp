#include <stan3/hmc_output_writers.hpp>
#include <stan3/arguments.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>

class HMCOutputWritersTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir = std::filesystem::temp_directory_path() / "stan3_hmc_test_output";
    std::filesystem::create_directories(test_dir);
    
    // Setup default test arguments
    args.output_dir = test_dir.string();
    args.num_chains = 2;
    args.save_start_params = false;
    args.save_warmup = false;
    args.save_diagnostics = false;
    args.save_metric = false;
  }
  
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
  }
  
  std::filesystem::path test_dir;
  stan3::hmc_nuts_args args;
};

TEST_F(HMCOutputWritersTest, CreateSingleChainWritersMinimal) {
  std::string model_name = "test_model";
  std::string timestamp = "20250522_143000";
  unsigned int chain_id = 1;
  
  auto writers = stan3::create_hmc_nuts_single_chain_writers(
    args, model_name, timestamp, chain_id);
  
  // Sample writer should always be created
  EXPECT_TRUE(writers.sample_writer != nullptr);
  
  // Optional writers should be nullptr when flags are false
  EXPECT_TRUE(writers.start_params_writer == nullptr);
  EXPECT_TRUE(writers.diagnostics_writer == nullptr);
  EXPECT_TRUE(writers.metric_writer == nullptr);
}

TEST_F(HMCOutputWritersTest, CreateSingleChainWritersAllOptions) {
  args.save_start_params = true;
  args.save_diagnostics = true;
  args.save_metric = true;
  
  std::string model_name = "test_model";
  std::string timestamp = "20250522_143000";
  unsigned int chain_id = 1;
  
  auto writers = stan3::create_hmc_nuts_single_chain_writers(
    args, model_name, timestamp, chain_id);
  
  // All writers should be created
  EXPECT_TRUE(writers.sample_writer != nullptr);
  EXPECT_TRUE(writers.start_params_writer != nullptr);
  EXPECT_TRUE(writers.diagnostics_writer != nullptr);
  EXPECT_TRUE(writers.metric_writer != nullptr);
}

TEST_F(HMCOutputWritersTest, CreateSingleChainWritersPartialOptions) {
  args.save_start_params = true;
  args.save_metric = true;
  // save_diagnostics remains false
  
  std::string model_name = "test_model";
  std::string timestamp = "20250522_143000";
  unsigned int chain_id = 1;
  
  auto writers = stan3::create_hmc_nuts_single_chain_writers(
    args, model_name, timestamp, chain_id);
  
  EXPECT_TRUE(writers.sample_writer != nullptr);
  EXPECT_TRUE(writers.start_params_writer != nullptr);
  EXPECT_TRUE(writers.diagnostics_writer == nullptr);
  EXPECT_TRUE(writers.metric_writer != nullptr);
}

TEST_F(HMCOutputWritersTest, CreateSingleChainWritersFilesCreated) {
  args.save_start_params = true;
  args.save_diagnostics = true;
  args.save_metric = true;
  
  std::string model_name = "test_model";
  std::string timestamp = "20250522_143000";
  unsigned int chain_id = 1;
  
  auto writers = stan3::create_hmc_nuts_single_chain_writers(
    args, model_name, timestamp, chain_id);
  
  // Write some test data to verify files are created
  std::vector<std::string> headers = {"param1", "param2"};
  std::vector<double> values = {1.5, 2.5};
  
  writers.sample_writer->operator()(headers);
  writers.sample_writer->operator()(values);
  
  writers.diagnostics_writer->operator()(headers);
  writers.diagnostics_writer->operator()(values);
  
  std::vector<std::string> param_names = {"param1", "param2"};
  std::vector<double> param_values = {1.0, 2.0};
  
  writers.start_params_writer->operator()(headers);
  writers.start_params_writer->operator()(values);

  writers.metric_writer->begin_record();
  writers.metric_writer->write("metric_type", "DIAG_E");
  writers.metric_writer->end_record();
  
  // Close writers to flush files
  writers.sample_writer.reset();
  writers.diagnostics_writer.reset();
  writers.start_params_writer.reset();
  writers.metric_writer.reset();
  
  // Check that expected files exist
  std::string sample_file = stan3::create_file_path(
    test_dir.string(),
    stan3::generate_filename(model_name, timestamp, chain_id, "sample", ".csv")
  );
  EXPECT_TRUE(std::filesystem::exists(sample_file));
  
  std::string diag_file = stan3::create_file_path(
    test_dir.string(),
    stan3::generate_filename(model_name, timestamp, chain_id, "param_grads", ".csv")
  );
  EXPECT_TRUE(std::filesystem::exists(diag_file));
  
  std::string start_file = stan3::create_file_path(
    test_dir.string(),
    stan3::generate_filename(model_name, timestamp, chain_id, "start_params", ".csv")
  );
  EXPECT_TRUE(std::filesystem::exists(start_file));
  
  std::string metric_file = stan3::create_file_path(
    test_dir.string(),
    stan3::generate_filename(model_name, timestamp, chain_id, "metric", ".json")
  );
  EXPECT_TRUE(std::filesystem::exists(metric_file));
}

TEST_F(HMCOutputWritersTest, CreateMultiChainWritersDefault) {
  args.num_chains = 3;
  std::string model_name = "multi_test";
  
  auto multi_writers = stan3::create_hmc_nuts_multi_chain_writers(
    args, model_name);
  
  // Should create writers for all chains
  EXPECT_EQ(multi_writers.size(), 3);
  
  // Each chain should have at least a sample writer
  for (size_t i = 0; i < multi_writers.size(); ++i) {
    EXPECT_TRUE(multi_writers[i].sample_writer != nullptr);
    
    // Optional writers should be nullptr when flags are false
    EXPECT_TRUE(multi_writers[i].start_params_writer == nullptr);
    EXPECT_TRUE(multi_writers[i].diagnostics_writer == nullptr);
    EXPECT_TRUE(multi_writers[i].metric_writer == nullptr);
  }
}

TEST_F(HMCOutputWritersTest, CreateMultiChainWritersAllOptions) {
  args.num_chains = 2;
  args.save_start_params = true;
  args.save_diagnostics = true;
  args.save_metric = true;
  
  std::string model_name = "multi_test";
  
  auto multi_writers = stan3::create_hmc_nuts_multi_chain_writers(
    args, model_name);
  
  EXPECT_EQ(multi_writers.size(), 2);
  
  // Each chain should have all writers
  for (size_t i = 0; i < multi_writers.size(); ++i) {
    EXPECT_TRUE(multi_writers[i].sample_writer != nullptr);
    EXPECT_TRUE(multi_writers[i].start_params_writer != nullptr);
    EXPECT_TRUE(multi_writers[i].diagnostics_writer != nullptr);
    EXPECT_TRUE(multi_writers[i].metric_writer != nullptr);
  }
}

TEST_F(HMCOutputWritersTest, CreateMultiChainWritersSingleChain) {
  args.num_chains = 1;
  std::string model_name = "single_chain_multi";
  
  auto multi_writers = stan3::create_hmc_nuts_multi_chain_writers(
    args, model_name);
  
  EXPECT_EQ(multi_writers.size(), 1);
  EXPECT_TRUE(multi_writers[0].sample_writer != nullptr);
}

TEST_F(HMCOutputWritersTest, CreateMultiChainWritersFilesCreated) {
  args.num_chains = 2;
  args.save_start_params = true;
  args.save_metric = true;
  
  std::string model_name = "file_test";
  
  auto multi_writers = stan3::create_hmc_nuts_multi_chain_writers(
    args, model_name);
  
  // Write test data for each chain
  std::vector<std::string> headers = {"param1"};
  
  for (size_t i = 0; i < multi_writers.size(); ++i) {
    std::vector<double> values = {static_cast<double>(i + 1)};
    multi_writers[i].sample_writer->operator()(headers);
    multi_writers[i].sample_writer->operator()(values);
    
    multi_writers[i].start_params_writer->operator()(headers);
    multi_writers[i].start_params_writer->operator()(values);
  }
  
  // Close all writers
  for (auto& writers : multi_writers) {
    writers.sample_writer.reset();
    writers.start_params_writer.reset();
  }
  
  // Check that files for each chain were created
  // Note: We can't predict the exact timestamp, so we check directory contents
  std::vector<std::filesystem::path> csv_files;
  std::vector<std::filesystem::path> json_files;
  
  for (const auto& entry : std::filesystem::directory_iterator(test_dir)) {
    if (entry.path().extension() == ".csv") {
      csv_files.push_back(entry.path());
    } else if (entry.path().extension() == ".json") {
      json_files.push_back(entry.path());
    }
  }
  
  // Should have 4 CSV files (2 x sample, start_params)
  EXPECT_EQ(csv_files.size(), 4);
  
  // Should have 2 JSON files (2 x metric)
  EXPECT_EQ(json_files.size(), 2);
  
  // Check that files contain expected chain identifiers
  bool found_chain1 = false, found_chain2 = false;
  for (const auto& file : csv_files) {
    std::string filename = file.filename().string();
    if (filename.find("chain1") != std::string::npos) {
      found_chain1 = true;
    }
    if (filename.find("chain2") != std::string::npos) {
      found_chain2 = true;
    }
  }
  EXPECT_TRUE(found_chain1);
  EXPECT_TRUE(found_chain2);
}

TEST_F(HMCOutputWritersTest, CreateWritersCustomCommentPrefix) {
  std::string model_name = "comment_test";
  std::string timestamp = "20250522_143000";
  std::string custom_prefix = "## ";
  
  auto writers = stan3::create_hmc_nuts_single_chain_writers(
    args, model_name, timestamp, 1, custom_prefix);
  
  // Write test data
  std::string comment = "comment";
  std::vector<double> values = {1.0};
  
  writers.sample_writer->operator()(comment);
  writers.sample_writer->operator()(values);
  
  writers.sample_writer.reset();
  
  // Check file content has custom prefix
  std::string sample_file = stan3::create_file_path(
    test_dir.string(),
    stan3::generate_filename(model_name, timestamp, 1, "sample", ".csv")
  );
  
  std::ifstream file(sample_file);
  std::string line;
  std::getline(file, line);
  
  EXPECT_EQ(line, "## comment");
}

TEST_F(HMCOutputWritersTest, CreateWritersInvalidOutputDir) {
  args.output_dir = "/root/invalid_directory_that_should_not_exist";
  
  std::string model_name = "error_test";
  
  // Should throw when trying to create writers in invalid directory
  EXPECT_THROW(
    stan3::create_hmc_nuts_multi_chain_writers(args, model_name),
    std::runtime_error
  );
}

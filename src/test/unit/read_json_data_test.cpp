#include <stan3/read_json_data.hpp>
#include <string>
#include <vector>
#include <gtest/gtest.h>

TEST(ReadJsonDataTest, HandlesValidJsonFile) {
    // Test that a valid JSON file is correctly read
    auto context = stan3::read_json_data("src/test/unit/json/valid_data.json");
    
    // Check that the context contains expected variables
    EXPECT_TRUE(context->contains_r("n"));
    EXPECT_TRUE(context->contains_r("x"));
    EXPECT_TRUE(context->contains_i("m"));
    
    // Check values
    EXPECT_EQ(context->vals_r("n")[0], 5.0);
    
    // Check array values
    std::vector<double> expected_x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_EQ(context->vals_r("x"), expected_x);
    
    // Check integer array values
    std::vector<int> expected_m = {1, 2, 3};
    EXPECT_EQ(context->vals_i("m"), expected_m);
    
    // Check dimensions
    EXPECT_EQ(context->dims_r("x").size(), 1);
    EXPECT_EQ(context->dims_r("x")[0], 5);
}

TEST(ReadJsonDataTest, HandlesEmptyFilename) {
    // Test that an empty filename returns an empty context
    auto context = stan3::read_json_data("");
    
    // Check that the context is empty
    std::vector<std::string> names;
    context->names_r(names);
    EXPECT_TRUE(names.empty());
    
    context->names_i(names);
    EXPECT_TRUE(names.empty());
}

TEST(ReadJsonDataTest, ThrowsOnNonexistentFile) {
    // Test that a nonexistent file throws an exception
    EXPECT_THROW(stan3::read_json_data("json/nonexistent_file.json"), std::runtime_error);
}

TEST(ReadJsonDataTest, ThrowsOnInvalidJson) {
    // Test that an invalid JSON file throws an exception
    EXPECT_THROW(stan3::read_json_data("src/test/unit/json/invalid_data.json"), stan::json::json_error);
}

TEST(ReadJsonDataTest, HandlesEmptyFile) {
    // Test that an empty file throws an appropriate exception
    EXPECT_THROW(stan3::read_json_data("src/test/unit/json/empty_data.json"), stan::json::json_error);
}












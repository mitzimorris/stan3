# Stan3 Unit Tests

## Test Files Created

### Core Tests
- `src/test/unit/load_model_test.cpp` - Tests for `load_model.hpp`
- `src/test/unit/load_samplers_test.cpp` - Tests for `load_samplers.hpp`

### Test Data Files
- `src/test/unit/test-models/bernoulli.data.json` - Bernoulli model test data
- `src/test/unit/json/valid_data.json` - Valid JSON data for testing
- `src/test/unit/json/invalid_data.json` - Invalid JSON for error testing

## Running the Tests

### Compile and Run Individual Tests
```bash
# Run load_model tests
./runTests.py src/test/unit/load_model_test.cpp

# Run load_samplers tests  
./runTests.py src/test/unit/load_samplers_test.cpp

# Run all unit tests
./runTests.py src/test/unit/
```

### Compile Only (without running)
```bash
# Compile individual test
make test/unit/load_model_test

# Compile multiple tests
make test/unit/load_model_test test/unit/load_samplers_test
```

### Run Compiled Tests Directly
```bash
# After compilation, run directly
./test/unit/load_model_test
./test/unit/load_samplers_test
```

## Test Coverage

### `load_model_test.cpp`
- ✅ Loading model with valid data
- ✅ Loading model with empty data context
- ✅ Different random seeds
- ✅ Consistent seed behavior
- ✅ Parameter count verification
- ✅ Data access verification
- ✅ Model name consistency
- ✅ Bernoulli-specific tests (if test files available)

### `load_samplers_test.cpp`
- ✅ All three metric types (UNIT_E, DIAG_E, DENSE_E)
- ✅ Single and multiple chains
- ✅ Variant system functionality
- ✅ Empty init contexts
- ✅ Null init writers
- ✅ Sampler parameter configuration
- ✅ Error handling for invalid metric types
- ✅ RNG seed consistency
- ✅ Bernoulli model integration (if test files available)

## Dependencies

Tests require:
- Google Test framework (included in Stan Math)
- Stan3 headers and compiled model
- TBB for parallel execution
- Filesystem support (C++17)

## File Structure
```
src/test/unit/
├── load_model_test.cpp
├── load_samplers_test.cpp
├── json/
│   ├── valid_data.json
│   └── invalid_data.json
└── test-models/
    └── bernoulli.data.json
```

## Notes

- Tests automatically skip Bernoulli-specific tests if model files aren't found
- Tests create temporary directories for file operations
- All temporary files are cleaned up automatically
- Tests verify both success cases and error handling
- Memory management is tested (unique_ptr usage)
- Thread safety is inherently tested through TBB usage

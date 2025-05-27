# Stan3

A next-generation interface for Stan that provides both command-line and programmatic access to Bayesian inference algorithms.

## Features

### Command-Line Interface

Stan3 provides a modern CLI built with [CLI11](https://github.com/CLIUtils/CLI11) that offers:

- **Clean argument parsing** with comprehensive validation and error messages
- **Flexible configuration** supporting all Stan inference parameters
- **Extensible design** for adding new algorithms beyond HMC-NUTS

**Usage:**
```bash
# Compile Stan model
make bernoulli_model.so

# Run HMC-NUTS sampling
./bernoulli hmc --data=data.json --chains=4 --samples=2000 --warmup=1000
```

### Current Implementation

- **HMC-NUTS Algorithm**: Full implementation of Stan's Hamiltonian Monte Carlo with No-U-Turn Sampler
- **Multiple Metrics**: Support for unit, diagonal, and dense mass matrices
- **Comprehensive Output**: Samples, diagnostics, initial values, and adapted metrics
- **Multi-chain Support**: Sequential execution with per-chain configuration

### Extensible Architecture

Stan3 is designed to easily accommodate new inference methods:

- Modular algorithm structure separating argument parsing, model loading, and execution
- Template-based sampler configuration supporting different metric types
- Consistent I/O patterns for all inference algorithms

**Planned algorithms**: Pathfinder, ADVI, MLE, Generate Quantities

## C API for Language Bindings

Stan3 includes a C API that enables integration with Python, Julia, R, and other languages. The key advantage is **stateful model loading** - load a model once, then run multiple inference algorithms without recompilation.

### Python Example

```python
import ctypes

# Load compiled model shared library
lib = ctypes.CDLL('./bernoulli_model.so')
error_buf = ctypes.create_string_buffer(1024)

# Load model once
model_args = ['stan3', '--data', 'data.json', '--seed', '42']
argc = len(model_args)
argv = (ctypes.c_char_p * argc)(*[arg.encode() for arg in model_args])

result = lib.stan3_load_model(argc, argv, error_buf, 1024)
if result != 0:
    raise RuntimeError(f"Load failed: {error_buf.value.decode()}")

# Run inference multiple times with different parameters
sampler_args = ['stan3', '--chains', '4', '--samples', '1000']
argc = len(sampler_args)
argv = (ctypes.c_char_p * argc)(*[arg.encode() for arg in sampler_args])

result = lib.stan3_run_samplers(argc, argv, error_buf, 1024)
if result != 0:
    raise RuntimeError(f"Sampling failed: {error_buf.value.decode()}")
```

### C API Functions

```c
// Load model with CLI arguments
int stan3_load_model(int argc, char** argv, char* error_message, size_t error_size);

// Run inference algorithms
int stan3_run_samplers(int argc, char** argv, char* error_message, size_t error_size);

// Utility functions
const char* stan3_get_model_name(void);
int stan3_is_model_loaded(void);
```

## Building

```bash
# Build shared library with model
make model_name_model.so

# Clean build artifacts
make clean-shared
```

## Benefits

- **Stateful Models**: Load once, sample many times without recompilation overhead
- **Argument Validation**: All CLI11 validation and error handling available to language bindings
- **Consistent Interface**: Same argument patterns across command-line and programmatic usage
- **Performance**: Compiled models with optimized Stan Math library
- **Extensibility**: Easy to add new algorithms and language bindings

## Status

Stan3 is under active development. The HMC-NUTS implementation is feature-complete and the C API enables basic Python integration. Additional algorithms and language bindings are planned.

---

Built on [Stan](https://mc-stan.org/) and [CLI11](https://github.com/CLIUtils/CLI11).

import ctypes

# Load the compiled model shared library
lib = ctypes.CDLL('./bernoulli_model.so')

# Create error buffer for error messages
error_buf = ctypes.create_string_buffer(1024)

# Load model with data
model_args = ['stan3', '--data', 'bernoulli.data.json', '--seed', '42']
argc = len(model_args)
argv = (ctypes.c_char_p * argc)(*[arg.encode() for arg in model_args])

result = lib.stan3_load_model(argc, argv, error_buf, 1024)
if result != 0:
    raise RuntimeError(f"Load failed: {error_buf.value.decode()}")

print("Model loaded successfully!")

# Run sampling
sampler_args = ['stan3', '--chains', '4', '--samples', '1000', '--warmup', '500']
argc = len(sampler_args)
argv = (ctypes.c_char_p * argc)(*[arg.encode() for arg in sampler_args])

result = lib.stan3_run_samplers(argc, argv, error_buf, 1024)
if result != 0:
    raise RuntimeError(f"Sampling failed: {error_buf.value.decode()}")

print("Sampling completed successfully!")

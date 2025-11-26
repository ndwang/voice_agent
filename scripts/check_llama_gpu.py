from llama_cpp import Llama, llama_backend_init, llama_supports_gpu_offload

# 1. Check if the library was compiled with GPU support
print(f"GPU Support Available: {llama_supports_gpu_offload()}")

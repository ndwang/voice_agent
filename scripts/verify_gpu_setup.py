#!/usr/bin/env python3
"""
Verify GPU Setup for PyTorch

Checks if PyTorch is installed with CUDA support and can detect GPUs.
Usage: python verify_gpu_setup.py
"""
import sys

def check_torch_cuda():
    """Check PyTorch CUDA installation."""
    print("=" * 60)
    print("PyTorch GPU Compatibility Check")
    print("=" * 60)
    print()
    
    # Check if torch is installed
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch is not installed")
        print("  Install with: uv sync")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✓ Yes' if cuda_available else '✗ No'}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(3, 3).cuda()
            print(f"\n✓ Successfully created tensor on GPU")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n✗ Error creating tensor on GPU: {e}")
            return False
    else:
        print("\n⚠ PyTorch was installed without CUDA support")
        print("  This could mean:")
        print("  1. CUDA toolkit is not installed on your system")
        print("  2. PyTorch was installed from CPU-only index")
        print("  3. CUDA version mismatch")
        return False
    
    # Check torchaudio
    try:
        import torchaudio
        print(f"\n✓ torchaudio version: {torchaudio.__version__}")
        if cuda_available:
            print("  torchaudio can use CUDA")
    except ImportError:
        print("\n⚠ torchaudio is not installed")
    
    print()
    print("=" * 60)
    print("GPU Setup: ✓ PASSED" if cuda_available else "GPU Setup: ✗ FAILED")
    print("=" * 60)
    
    return cuda_available

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed on the system."""
    import subprocess
    
    print("\n" + "=" * 60)
    print("System CUDA Toolkit Check")
    print("=" * 60)
    print()
    
    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ NVIDIA drivers detected")
            # Extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print("✗ nvidia-smi not available")
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  NVIDIA drivers may not be installed")
    except subprocess.TimeoutExpired:
        print("⚠ nvidia-smi timed out")
    except Exception as e:
        print(f"⚠ Error checking nvidia-smi: {e}")

def main():
    """Main function."""
    torch_ok = check_torch_cuda()
    check_cuda_toolkit()
    
    print()
    if torch_ok:
        print("Recommendation: Your PyTorch installation has GPU support!")
    else:
        print("Recommendation:")
        print("1. Verify CUDA toolkit is installed (check nvidia-smi output above)")
        print("2. Reinstall PyTorch with: uv sync --reinstall-package torch torchaudio")
        print("3. Ensure workspace pyproject.toml has CUDA configuration")
        print("4. Check CUDA version matches (currently configured for CUDA 12.8)")
    
    return 0 if torch_ok else 1

if __name__ == "__main__":
    sys.exit(main())


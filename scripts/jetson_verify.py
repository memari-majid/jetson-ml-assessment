#!/usr/bin/env python3
"""
NVIDIA Jetson System Verification Script
Checks system configuration, GPU access, and installed frameworks
"""

import sys
import os
import subprocess
import importlib

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_command(cmd):
    """Check if a command exists"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0, result.stdout.strip()
    except:
        return False, ""

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError:
        return False, None

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NVIDIA Jetson Orin Nano - System Verification Tool         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    issues_found = []
    warnings = []
    
    # System Information
    print_section("System Information")
    
    # OS Info
    exists, output = check_command("lsb_release -d")
    if exists:
        print(f"✅ OS: {output.split(':')[1].strip() if ':' in output else output}")
    else:
        print("⚠️  OS: Unable to determine")
        warnings.append("Could not determine OS version")
    
    # Kernel
    exists, output = check_command("uname -r")
    if exists:
        print(f"✅ Kernel: {output}")
    
    # Architecture
    exists, output = check_command("uname -m")
    if exists:
        print(f"✅ Architecture: {output}")
    
    # Python version
    print(f"✅ Python: {sys.version.split()[0]}")
    
    # GPU and CUDA
    print_section("GPU and CUDA")
    
    # Check nvidia-smi
    exists, output = check_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if exists and output:
        print(f"✅ GPU Detected: {output}")
    else:
        print("❌ GPU: Not detected by nvidia-smi")
        issues_found.append("GPU not detected - driver issue?")
    
    # Check CUDA
    exists, output = check_command("nvcc --version | grep 'release'")
    if exists and output:
        cuda_ver = output.split("release")[1].split(",")[0].strip()
        print(f"✅ CUDA Compiler: {cuda_ver}")
    else:
        print("⚠️  CUDA Compiler: Not found in PATH")
        warnings.append("CUDA compiler (nvcc) not in PATH")
    
    # Check CUDA path
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✅ CUDA_HOME: {cuda_path}")
    else:
        print("⚠️  CUDA_HOME: Not set")
        warnings.append("CUDA_HOME environment variable not set")
    
    # Check JetPack
    exists, output = check_command("dpkg -l | grep nvidia-jetpack")
    if exists and output:
        print(f"✅ JetPack: Installed")
    else:
        print("⚠️  JetPack: Not detected")
        warnings.append("JetPack may not be fully installed")
    
    # Python ML Frameworks
    print_section("Python ML Frameworks")
    
    frameworks = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
    ]
    
    for import_name, display_name in frameworks:
        installed, version = check_python_package(import_name, import_name)
        if installed:
            print(f"✅ {display_name}: {version}")
        else:
            print(f"❌ {display_name}: Not installed")
            issues_found.append(f"{display_name} not installed")
    
    # PyTorch CUDA Check
    print_section("PyTorch CUDA Integration")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ PyTorch CUDA: Available")
            print(f"✅ CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"✅ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"❌ PyTorch CUDA: Not Available")
            print(f"   PyTorch Version: {torch.__version__}")
            if '+cpu' in torch.__version__:
                print(f"   ⚠️  PyTorch is CPU-only version")
                issues_found.append("PyTorch installed without CUDA support")
            else:
                issues_found.append("PyTorch cannot access CUDA (driver/config issue)")
    except ImportError:
        print("❌ PyTorch: Not installed")
        issues_found.append("PyTorch not installed")
    
    # TensorRT Check
    print_section("TensorRT")
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT: {trt.__version__}")
    except ImportError:
        print("⚠️  TensorRT: Not installed")
        warnings.append("TensorRT not installed (optional but recommended)")
    
    # System Resources
    print_section("System Resources")
    
    try:
        import psutil
        
        cpu_count = psutil.cpu_count()
        mem = psutil.virtual_memory()
        mem_total = mem.total / (1024**3)
        mem_avail = mem.available / (1024**3)
        
        print(f"✅ CPU Cores: {cpu_count}")
        print(f"✅ Total RAM: {mem_total:.2f} GB")
        print(f"✅ Available RAM: {mem_avail:.2f} GB")
        
        if mem_avail < 1.0:
            warnings.append(f"Low available memory: {mem_avail:.2f} GB")
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_free = disk.free / (1024**3)
        print(f"✅ Free Disk Space: {disk_free:.2f} GB")
        
        if disk_free < 10:
            warnings.append(f"Low disk space: {disk_free:.2f} GB")
            
    except ImportError:
        print("⚠️  psutil not installed - cannot check system resources")
    
    # Benchmark Scripts
    print_section("Benchmark Scripts")
    
    scripts = [
        'jetson_simple_benchmark.py',
        'jetson_ml_benchmark.py',
        'jetson_gpu_benchmark.py',
        'jetson_verify.py',
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}: Found")
        else:
            print(f"⚠️  {script}: Not found")
    
    # Summary
    print_section("Summary")
    
    if not issues_found and not warnings:
        print("✅ All checks passed! System is ready for ML workloads.")
    else:
        if issues_found:
            print(f"\n❌ Issues Found ({len(issues_found)}):")
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
    
    # Recommendations
    print_section("Recommendations")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n🔴 CRITICAL: GPU not accessible")
            print("   Action: Follow NEXT_STEPS_PLAN.md Phase 1")
            print("   1. Install PyTorch with CUDA support")
            print("   2. Verify GPU drivers with: nvidia-smi")
            print("   3. Set environment variables (CUDA_HOME, PATH)")
    except:
        pass
    
    if warnings and not issues_found:
        print("\n🟡 System is functional but can be optimized")
        print("   Consider: Installing recommended packages")
        print("   See: requirements.txt and SETUP_GUIDE.md")
    
    if not issues_found and not warnings:
        print("\n🟢 System is fully configured and ready")
        print("   Next: Run benchmark scripts")
        print("   - CPU tests: python3 jetson_simple_benchmark.py")
        print("   - GPU tests: python3 jetson_gpu_benchmark.py")
    
    print("\n" + "="*60)
    print(f"Verification complete: {len(issues_found)} issues, {len(warnings)} warnings")
    print("="*60 + "\n")
    
    # Return exit code
    return 1 if issues_found else 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


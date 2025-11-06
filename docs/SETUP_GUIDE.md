# Setup Guide - NVIDIA Jetson Orin Nano ML Environment

**Last Updated:** October 14, 2025  
**Platform:** NVIDIA Jetson Orin Nano  
**OS:** Ubuntu 22.04.5 LTS

---

## Overview

This guide documents the complete setup process for the machine learning environment on the NVIDIA Jetson Orin Nano, including all packages installed and configuration steps performed during the assessment.

---

## System Information

### Hardware Specifications
- **Model:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- **CPU:** 6-core ARM Cortex-A78AE @ 1.728 GHz
- **GPU:** NVIDIA Orin (Ampere architecture)
- **Memory:** 7.4GB RAM
- **Storage:** 467GB NVMe SSD
- **Architecture:** ARM64 (aarch64)

### Software Baseline
- **OS:** Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel:** 5.15.148-tegra
- **JetPack:** R36.4.7
- **CUDA:** 12.6 (installed but needs configuration)
- **Python:** 3.10.12

---

## Installation Steps Performed

### 1. System Update
```bash
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential cmake git wget curl
```

**Packages Installed:**
- python3-pip (22.0.2+dfsg-1ubuntu0.7)
- python3-dev (3.10.6-1~22.04.1)
- build-essential (12.9ubuntu3)
- cmake (3.22.1-1ubuntu1.22.04.2)
- git (2.34.1-1ubuntu1.15)
- wget (1.21.2-2ubuntu1.1)
- curl (7.81.0-1ubuntu1.21)

### 2. PyTorch Installation
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**PyTorch Stack:**
- torch: 2.9.0+cpu
- torchvision: 0.24.0
- torchaudio: 2.9.0
- Dependencies: filelock, fsspec, jinja2, mpmath, networkx, sympy, typing-extensions

**Note:** CPU-only version installed. For GPU support, CUDA-enabled PyTorch required.

### 3. TensorFlow Installation
```bash
pip3 install tensorflow opencv-python matplotlib seaborn pandas scikit-learn
```

**ML Frameworks:**
- tensorflow: 2.20.0
- keras: 3.11.3
- opencv-python: 4.9.0.80
- scikit-learn: 1.7.2

**Supporting Libraries:**
- numpy: 2.2.6 (downgraded from initial install for compatibility)
- pandas: 1.3.5 (system package)
- seaborn: 0.13.2
- matplotlib: 3.5.1 (system package)

### 4. System Monitoring Tools
```bash
pip3 install psutil GPUtil
```

**Monitoring Tools:**
- psutil: 7.1.0
- GPUtil: 1.4.0

### 5. NVIDIA Benchmarking Tools
```bash
git clone https://github.com/NVIDIA-AI-IOT/jetson_benchmarks.git
```

**Repository:** NVIDIA AI IOT Jetson Benchmarks  
**Location:** /home/mj/jetson_benchmarks

---

## Complete Package List

### Python Packages (pip3 list)
```
Package                  Version
----------------------- ----------------
absl-py                  2.3.1
astunparse               1.6.3
filelock                 3.19.1
flatbuffers              25.9.23
fsspec                   2025.9.0
gast                     0.6.0
google_pasta             0.2.0
GPUtil                   1.4.0
grpcio                   1.75.1
h5py                     3.15.0
jinja2                   3.1.6
joblib                   1.5.2
keras                    3.11.3
libclang                 18.1.1
markdown                 3.9
markdown-it-py           4.0.0
MarkupSafe               3.0.3
mdurl                    0.1.2
ml_dtypes                0.5.3
mpmath                   1.3.0
namex                    0.1.0
networkx                 3.3
numpy                    2.2.6
opencv-python            4.9.0.80
opt_einsum               3.4.0
optree                   0.17.0
protobuf                 6.32.1
psutil                   7.1.0
pygments                 2.19.2
rich                     14.2.0
scikit-learn             1.7.2
seaborn                  0.13.2
sympy                    1.13.3
tensorboard              2.20.0
tensorboard-data-server  0.7.2
tensorflow               2.20.0
termcolor                3.1.0
threadpoolctl            3.6.0
torch                    2.9.0+cpu
torchaudio               2.9.0
torchvision              0.24.0
typing_extensions        4.15.0
werkzeug                 3.1.3
wrapt                    1.17.3
```

### System Packages (relevant)
- matplotlib (3.5.1)
- numpy (initially 1.21.5, upgraded via pip)
- pandas (1.3.5)
- scipy (1.8.0)
- pillow (9.0.1)

---

## Directory Structure Created

```
/home/mj/
├── jetson_ml_assessment_2025-10-14/
│   ├── EXECUTIVE_SUMMARY.md
│   ├── NVIDIA_Jetson_Orin_Nano_ML_Comprehensive_Report.md
│   ├── SETUP_GUIDE.md
│   ├── NEXT_STEPS_PLAN.md
│   ├── jetson_simple_benchmark.py
│   ├── jetson_ml_benchmark.py
│   └── jetson_benchmark_results.json
└── jetson_benchmarks/
    ├── benchmark.py
    ├── benchmark_csv/
    ├── utils/
    └── README.md
```

---

## Configuration Files

### CUDA Environment
**Location:** /usr/local/cuda-12.6/

**Key Directories:**
- bin/ - CUDA compiler and tools
- lib64/ - CUDA libraries
- include/ - CUDA headers
- nvvm/ - NVVM compiler
- targets/ - Target-specific files

**Environment Variables (not set, needs configuration):**
```bash
# Add to ~/.bashrc for CUDA access
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

---

## Known Issues and Workarounds

### Issue 1: PyTorch CUDA Not Available
**Problem:** `torch.cuda.is_available()` returns False  
**Cause:** PyTorch installed from CPU-only wheel  
**Solution:** Requires PyTorch built with CUDA support for Jetson

### Issue 2: NumPy Version Conflicts
**Problem:** TensorFlow requires numpy<2, OpenCV requires numpy>=2  
**Workaround:** Installed numpy 2.2.6 with opencv-python 4.9.0.80  
**Status:** Functional but with warnings

### Issue 3: Matplotlib Compatibility
**Problem:** Matplotlib conflicts with NumPy 2.x on system packages  
**Workaround:** Disabled visualization in benchmark scripts  
**Alternative:** Use headless plotting or upgrade matplotlib via pip

### Issue 4: GPU Not Detected by ML Frameworks
**Problem:** nvidia-smi shows GPU but PyTorch can't access it  
**Cause:** Missing CUDA-enabled PyTorch build for Jetson  
**Priority:** HIGH - See NEXT_STEPS_PLAN.md

---

## Verification Commands

### Check System Info
```bash
# CPU info
lscpu

# Memory info
free -h

# GPU info
nvidia-smi

# CUDA version
nvcc --version

# Python version
python3 --version
```

### Check Installed Packages
```bash
# PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__)"

# TensorFlow
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

# OpenCV
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Run Benchmarks
```bash
cd /home/mj/jetson_ml_assessment_2025-10-14
python3 jetson_simple_benchmark.py
```

---

## Performance Baseline

### Established Benchmarks
- **ResNet-18 Inference:** 428.96 ± 81.06 ms (9.32 FPS)
- **MobileNet-v2 Inference:** 447.48 ± 14.17 ms (8.94 FPS)
- **Matrix 2000×2000:** 259.46 ms (61.67 GFLOPS)
- **CPU Usage:** 22.3% average during ML tasks
- **Memory Usage:** 48.1% average during ML tasks

---

## Troubleshooting

### GPU Access Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
ls /usr/local/cuda-12.6/

# Check PyTorch build
python3 -c "import torch; print(torch.version.cuda)"
```

### Package Conflicts
```bash
# Check installed versions
pip3 list | grep -E "numpy|torch|tensorflow"

# Reinstall if needed
pip3 install --force-reinstall <package_name>
```

### Memory Issues
```bash
# Check available memory
free -h

# Monitor during execution
watch -n 1 free -h
```

---

## Backup and Restore

### Backup Current Environment
```bash
# Save pip packages
pip3 freeze > /home/mj/jetson_ml_assessment_2025-10-14/requirements.txt

# Backup scripts
tar -czf jetson_ml_backup.tar.gz /home/mj/jetson_ml_assessment_2025-10-14/
```

### Restore Environment
```bash
# Install from requirements
pip3 install -r requirements.txt

# Restore files
tar -xzf jetson_ml_backup.tar.gz
```

---

## Next Steps

See **NEXT_STEPS_PLAN.md** for:
1. GPU enablement procedures
2. Performance optimization strategies
3. Production deployment guidelines
4. Advanced configuration options

---

## References

- NVIDIA Jetson Documentation: https://developer.nvidia.com/embedded/jetson-orin-nano
- PyTorch for Jetson: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- JetPack SDK: https://developer.nvidia.com/embedded/jetpack

---

**Setup Documented By:** ML Assessment System  
**Environment Status:** Functional (CPU-only mode)  
**Last Tested:** October 14, 2025

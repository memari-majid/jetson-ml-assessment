# Jetson ML Benchmark Suite - Quick Start Guide

**Last Updated:** October 14, 2025

## 🚀 Get Started in 5 Minutes

### Step 1: Verify System
```bash
cd /home/mj/jetson_ml_assessment_2025-10-14
python3 jetson_verify.py
```

This will check:
- GPU availability
- CUDA installation
- Python packages
- System resources

### Step 2: Run CPU Benchmark
```bash
python3 jetson_simple_benchmark.py
```

**Expected Output:**
- Takes ~60 seconds to complete
- Tests 3 PyTorch models
- Generates `jetson_benchmark_results.json`
- Shows FPS performance metrics

### Step 3: Check Results
```bash
cat jetson_benchmark_results.json | grep throughput_fps
```

**Baseline Performance (CPU-only):**
- ResNet-18: ~9 FPS
- MobileNet-v2: ~9 FPS
- ResNet-50: ~3 FPS

### Step 4 (If GPU Available): Run GPU Benchmark
```bash
python3 jetson_gpu_benchmark.py
```

**Expected GPU Performance:**
- ResNet-18: ~50-70 FPS
- MobileNet-v2: ~80-100 FPS
- 5-10x speedup over CPU

---

## 📋 Complete Test Suite

### Run All Tests Automatically
```bash
python3 run_all_tests.py
```

**Options:**
```bash
# Skip GPU tests
python3 run_all_tests.py --skip-gpu

# Quick mode (faster, skips long tests)
python3 run_all_tests.py --quick
```

### Compare Results
```bash
# CPU-only analysis
python3 compare_results.py jetson_benchmark_results.json

# CPU vs GPU comparison
python3 compare_results.py jetson_benchmark_results.json jetson_gpu_benchmark_results.json
```

---

## 🔧 Available Scripts

| Script | Purpose | Runtime | GPU Required |
|--------|---------|---------|--------------|
| `jetson_verify.py` | System verification | 10 sec | No |
| `jetson_simple_benchmark.py` | CPU benchmark | 60 sec | No |
| `jetson_gpu_benchmark.py` | GPU benchmark | 90 sec | Yes |
| `run_all_tests.py` | Automated test runner | 3-5 min | No |
| `compare_results.py` | Results comparison | 5 sec | No |
| `tensorrt_optimizer.py` | TensorRT optimization | 5-10 min | Yes |

---

## ⚙️ Installation

### Install Dependencies
```bash
pip3 install -r requirements.txt
```

### GPU Enablement (If CUDA Not Working)
See **NEXT_STEPS_PLAN.md** Phase 1 for detailed instructions.

**Quick check:**
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If False:
1. Install PyTorch with CUDA support
2. Set environment variables
3. Verify with `nvidia-smi`

---

## 📊 Understanding Results

### CPU Benchmark Results
```json
{
  "pytorch_models": {
    "resnet18": {
      "throughput_fps": 9.32,
      "avg_inference_time": 0.429
    }
  }
}
```

**Key Metrics:**
- **throughput_fps**: Images processed per second
- **avg_inference_time**: Time per batch (seconds)
- **model_size_mb**: Model memory footprint

### GPU Benchmark Results
```json
{
  "pytorch_gpu": {
    "resnet18": {
      "batch_8": {
        "throughput_fps": 65.5,
        "latency_ms": 12.2
      }
    }
  }
}
```

**Key Metrics:**
- **throughput_fps**: Total throughput
- **latency_ms**: Per-image latency
- **memory_allocated_mb**: GPU memory used

---

## 🎯 Common Use Cases

### 1. Quick Performance Check
```bash
python3 jetson_verify.py && python3 jetson_simple_benchmark.py
```

### 2. Full System Evaluation
```bash
python3 run_all_tests.py
```

### 3. GPU Performance Testing
```bash
python3 jetson_gpu_benchmark.py
python3 compare_results.py jetson_benchmark_results.json jetson_gpu_benchmark_results.json
```

### 4. Model Optimization
```bash
python3 tensorrt_optimizer.py
```

---

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Solution:** See NEXT_STEPS_PLAN.md Phase 1

### Out of Memory
- Reduce batch size in scripts
- Close other applications
- Check available memory: `free -h`

### Slow Performance
- Verify GPU is being used
- Check power mode: `sudo nvpmodel -q`
- Enable max performance: `sudo nvpmodel -m 0`

---

## 📖 Documentation Structure

```
📁 jetson_ml_assessment_2025-10-14/
├── 📄 QUICK_START.md (this file) - Start here
├── 📄 README.md - Full overview
├── 📄 EXECUTIVE_SUMMARY.md - High-level findings
├── 📄 NEXT_STEPS_PLAN.md - Optimization roadmap
├── 📄 SETUP_GUIDE.md - Detailed installation
├── 🐍 jetson_verify.py - System check
├── 🐍 jetson_simple_benchmark.py - CPU tests
├── 🐍 jetson_gpu_benchmark.py - GPU tests
├── 🐍 run_all_tests.py - Automated runner
├── 🐍 compare_results.py - Analysis tool
└── 🐍 tensorrt_optimizer.py - Optimization
```

**Reading Order:**
1. QUICK_START.md (you are here)
2. Run `jetson_verify.py`
3. Review EXECUTIVE_SUMMARY.md
4. See NEXT_STEPS_PLAN.md for optimization

---

## ✅ Success Checklist

- [ ] Run `jetson_verify.py` - system check
- [ ] Run `jetson_simple_benchmark.py` - baseline
- [ ] Review `jetson_benchmark_results.json`
- [ ] Read EXECUTIVE_SUMMARY.md
- [ ] If GPU available: run `jetson_gpu_benchmark.py`
- [ ] If GPU not working: see NEXT_STEPS_PLAN.md
- [ ] Run `compare_results.py` for analysis

---

## 🎓 Next Steps

### For CPU-Only Systems
1. ✅ Baseline established
2. 🔴 Enable GPU (see NEXT_STEPS_PLAN.md)
3. 🟡 Optimize models
4. 🟢 Deploy to production

### For GPU-Enabled Systems
1. ✅ Full performance available
2. 🟠 Install TensorRT
3. 🟡 Quantize models (INT8)
4. 🟢 Production deployment

---

## 💡 Tips

- **First time?** Start with `jetson_verify.py`
- **Need speed?** Use `--quick` flag
- **No GPU?** CPU benchmarks still valuable
- **Production?** See NEXT_STEPS_PLAN.md Phase 3

---

## 📞 Support

**Issues?** Check:
1. This QUICK_START guide
2. SETUP_GUIDE.md for installation
3. NEXT_STEPS_PLAN.md for GPU issues
4. NVIDIA Developer Forums

**Documentation:**
- Full report: NVIDIA_Jetson_Orin_Nano_ML_Comprehensive_Report.md
- Setup: SETUP_GUIDE.md
- Planning: NEXT_STEPS_PLAN.md

---

**Status:** ✅ Ready to use  
**Platform:** NVIDIA Jetson Orin Nano  
**Last Tested:** October 14, 2025

---

Start benchmarking in 3 commands:
```bash
python3 jetson_verify.py
python3 jetson_simple_benchmark.py
python3 compare_results.py jetson_benchmark_results.json
```

Good luck! 🚀


#!/usr/bin/env python3
"""
NVIDIA Jetson Orin Nano GPU Benchmark Suite
Tests GPU-accelerated ML performance with CUDA and TensorRT optimizations
"""

import time
import json
import psutil
import numpy as np
from datetime import datetime
import sys
import torch
import torchvision
from torchvision import models

class JetsonGPUBenchmark:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        self.system_info = self.get_system_info()
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA not available. This script requires GPU access.")
            print("   Please run jetson_simple_benchmark.py for CPU-only tests.")
            print("   See NEXT_STEPS_PLAN.md for GPU enablement instructions.")
            sys.exit(1)
        
    def get_system_info(self):
        """Collect system information including GPU details"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            
            # Get GPU memory info
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_memory_total'] = gpu_mem
        
        return info
    
    def benchmark_pytorch_gpu(self):
        """Benchmark PyTorch models on GPU"""
        print("Running PyTorch GPU benchmarks...")
        
        device = torch.device('cuda')
        
        # Test different model sizes
        models_to_test = {
            'resnet18': models.resnet18(pretrained=False),
            'resnet50': models.resnet50(pretrained=False),
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
        }
        
        results = {}
        batch_sizes = [1, 4, 8, 16]  # Test different batch sizes
        input_size = (3, 224, 224)
        
        for model_name, model in models_to_test.items():
            print(f"  Testing {model_name}...")
            model = model.to(device)
            model.eval()
            
            results[model_name] = {}
            
            for batch_size in batch_sizes:
                print(f"    Batch size: {batch_size}")
                
                # Create dummy input on GPU
                dummy_input = torch.randn(batch_size, *input_size).to(device)
                
                # Warmup
                for _ in range(10):
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                # Synchronize GPU before timing
                torch.cuda.synchronize()
                
                # Benchmark inference
                times = []
                for _ in range(50):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(dummy_input)
                    torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / avg_time
                
                results[model_name][f'batch_{batch_size}'] = {
                    'avg_inference_time': avg_time,
                    'std_inference_time': std_time,
                    'throughput_fps': throughput,
                    'latency_ms': avg_time * 1000 / batch_size
                }
            
            # Add model metadata
            results[model_name]['model_params'] = sum(p.numel() for p in model.parameters())
            results[model_name]['model_size_mb'] = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024**2)
        
        return results
    
    def benchmark_mixed_precision(self):
        """Benchmark FP16 (half precision) performance"""
        print("Running mixed precision (FP16) benchmarks...")
        
        device = torch.device('cuda')
        model = models.resnet50(pretrained=False).to(device)
        model.eval()
        
        batch_size = 8
        input_size = (3, 224, 224)
        
        results = {}
        
        # FP32 baseline
        print("  Testing FP32...")
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        times_fp32 = []
        for _ in range(30):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()
            times_fp32.append(time.time() - start)
        
        results['fp32'] = {
            'avg_time': np.mean(times_fp32),
            'std_time': np.std(times_fp32),
            'throughput_fps': batch_size / np.mean(times_fp32)
        }
        
        # FP16 testing
        print("  Testing FP16...")
        model_fp16 = model.half()
        dummy_input_fp16 = dummy_input.half()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_fp16(dummy_input_fp16)
        
        torch.cuda.synchronize()
        times_fp16 = []
        for _ in range(30):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model_fp16(dummy_input_fp16)
            torch.cuda.synchronize()
            times_fp16.append(time.time() - start)
        
        results['fp16'] = {
            'avg_time': np.mean(times_fp16),
            'std_time': np.std(times_fp16),
            'throughput_fps': batch_size / np.mean(times_fp16)
        }
        
        # Calculate speedup
        results['fp16_speedup'] = results['fp32']['avg_time'] / results['fp16']['avg_time']
        
        return results
    
    def benchmark_gpu_memory(self):
        """Benchmark GPU memory usage for different models"""
        print("Running GPU memory benchmarks...")
        
        device = torch.device('cuda')
        results = {}
        
        models_to_test = {
            'resnet18': models.resnet18(pretrained=False),
            'resnet50': models.resnet50(pretrained=False),
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
        }
        
        for model_name, model in models_to_test.items():
            print(f"  Testing {model_name}...")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Load model to GPU
            model = model.to(device)
            model.eval()
            
            # Create input
            dummy_input = torch.randn(8, 3, 224, 224).to(device)
            
            # Run inference
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            
            results[model_name] = {
                'memory_allocated_mb': allocated,
                'memory_reserved_mb': reserved,
                'max_memory_allocated_mb': max_allocated
            }
            
            # Clean up
            del model
            del dummy_input
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_cuda_operations(self):
        """Benchmark CUDA tensor operations"""
        print("Running CUDA tensor operation benchmarks...")
        
        device = torch.device('cuda')
        results = {}
        
        # Matrix multiplication
        print("  Testing matrix multiplication...")
        matrix_sizes = [100, 500, 1000, 2000, 4000]
        
        for size in matrix_sizes:
            a = torch.randn(size, size).to(device)
            b = torch.randn(size, size).to(device)
            
            # Warmup
            for _ in range(5):
                _ = torch.mm(a, b)
            
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.time()
                result = torch.mm(a, b)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            flops = 2 * size**3
            gflops = (flops / avg_time) / 1e9
            
            results[f'matmul_{size}x{size}'] = {
                'avg_time': avg_time,
                'gflops': gflops
            }
        
        # Convolution operations
        print("  Testing convolution...")
        conv_configs = [
            (64, 128, 3, 224),  # in_channels, out_channels, kernel, size
            (128, 256, 3, 112),
            (256, 512, 3, 56),
        ]
        
        for in_ch, out_ch, kernel, size in conv_configs:
            conv = torch.nn.Conv2d(in_ch, out_ch, kernel, padding=1).to(device)
            input_tensor = torch.randn(4, in_ch, size, size).to(device)
            
            # Warmup
            for _ in range(5):
                _ = conv(input_tensor)
            
            torch.cuda.synchronize()
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.time()
                _ = conv(input_tensor)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            results[f'conv_{in_ch}_{out_ch}_{size}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times)
            }
        
        return results
    
    def monitor_gpu_utilization(self, duration=20):
        """Monitor GPU utilization during operations"""
        print("Monitoring GPU utilization...")
        
        device = torch.device('cuda')
        
        # Load a model to stress GPU
        model = models.resnet50(pretrained=False).to(device)
        model.eval()
        
        gpu_stats = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Run inference to keep GPU busy
            dummy_input = torch.randn(8, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Collect stats
            stats = {
                'timestamp': time.time() - start_time,
                'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2)
            }
            gpu_stats.append(stats)
            
            time.sleep(0.5)
        
        return {
            'stats': gpu_stats,
            'avg_memory_mb': np.mean([s['memory_allocated_mb'] for s in gpu_stats]),
            'max_memory_mb': np.max([s['memory_allocated_mb'] for s in gpu_stats])
        }
    
    def run_benchmark(self):
        """Run all GPU benchmarks"""
        print("Starting NVIDIA Jetson Orin Nano GPU Benchmark...")
        print(f"GPU: {self.system_info.get('gpu_name', 'Unknown')}")
        print(f"CUDA Version: {self.system_info.get('cuda_version', 'Unknown')}")
        
        self.results['system_info'] = self.system_info
        self.results['pytorch_gpu'] = self.benchmark_pytorch_gpu()
        self.results['mixed_precision'] = self.benchmark_mixed_precision()
        self.results['gpu_memory'] = self.benchmark_gpu_memory()
        self.results['cuda_operations'] = self.benchmark_cuda_operations()
        self.results['gpu_monitoring'] = self.monitor_gpu_utilization()
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.results['total_benchmark_time'] = total_time
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive GPU benchmark report"""
        print("\n" + "="*60)
        print("NVIDIA JETSON ORIN NANO GPU BENCHMARK REPORT")
        print("="*60)
        
        print(f"\nGPU Information:")
        print(f"  GPU: {self.system_info['gpu_name']}")
        print(f"  CUDA Version: {self.system_info['cuda_version']}")
        print(f"  GPU Memory: {self.system_info.get('gpu_memory_total', 'N/A'):.2f} GB")
        print(f"  PyTorch Version: {self.system_info['pytorch_version']}")
        
        print(f"\nPyTorch GPU Performance:")
        for model, metrics in self.results['pytorch_gpu'].items():
            print(f"  {model}:")
            for batch_key, batch_metrics in metrics.items():
                if batch_key.startswith('batch_'):
                    batch_size = batch_key.split('_')[1]
                    print(f"    Batch {batch_size}:")
                    print(f"      Throughput: {batch_metrics['throughput_fps']:.2f} FPS")
                    print(f"      Latency: {batch_metrics['latency_ms']:.2f} ms")
        
        print(f"\nMixed Precision Performance:")
        mp = self.results['mixed_precision']
        print(f"  FP32 Throughput: {mp['fp32']['throughput_fps']:.2f} FPS")
        print(f"  FP16 Throughput: {mp['fp16']['throughput_fps']:.2f} FPS")
        print(f"  FP16 Speedup: {mp['fp16_speedup']:.2f}x")
        
        print(f"\nGPU Memory Usage:")
        for model, mem in self.results['gpu_memory'].items():
            print(f"  {model}: {mem['max_memory_allocated_mb']:.2f} MB")
        
        print(f"\nCUDA Operations Performance:")
        for op, metrics in self.results['cuda_operations'].items():
            if 'gflops' in metrics:
                print(f"  {op}: {metrics['gflops']:.2f} GFLOPS")
        
        print(f"\nTotal Benchmark Time: {self.results['total_benchmark_time']:.1f} seconds")
        
        # Save results
        output_file = 'gb10_gpu_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    try:
        benchmark = JetsonGPUBenchmark()
        results = benchmark.run_benchmark()
        benchmark.generate_report()
        print("\n✅ GPU Benchmark completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf CUDA is not available, please:")
        print("  1. Check NEXT_STEPS_PLAN.md for GPU enablement instructions")
        print("  2. Run 'python3 jetson_verify.py' to check system status")
        print("  3. Use 'python3 jetson_simple_benchmark.py' for CPU-only tests")
        sys.exit(1)


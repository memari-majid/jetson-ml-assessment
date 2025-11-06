#!/usr/bin/env python3
"""
NVIDIA Jetson Orin Nano Simple ML Benchmark
Focus on PyTorch CPU performance testing
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

class SimpleJetsonBenchmark:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """Collect system information"""
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
        
        return info
    
    def benchmark_pytorch_models(self):
        """Benchmark PyTorch models on CPU"""
        print("Running PyTorch CPU benchmarks...")
        
        # Test different model sizes
        models_to_test = {
            'resnet18': models.resnet18(pretrained=False),
            'resnet50': models.resnet50(pretrained=False),
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
        }
        
        results = {}
        batch_size = 4  # Smaller batch size for Jetson
        input_size = (3, 224, 224)
        
        for model_name, model in models_to_test.items():
            print(f"  Testing {model_name}...")
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_size)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark inference
            times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    output = model(dummy_input)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            results[model_name] = {
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'throughput_fps': throughput,
                'model_params': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            }
        
        return results
    
    def benchmark_matrix_operations(self):
        """Benchmark matrix operations"""
        print("Running matrix operation benchmarks...")
        
        results = {}
        
        # Test different matrix sizes
        matrix_sizes = [100, 500, 1000, 2000]
        
        for size in matrix_sizes:
            print(f"  Testing {size}x{size} matrices...")
            
            # Create random matrices
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            # Matrix multiplication benchmark
            times = []
            for _ in range(5):
                start_time = time.time()
                result = torch.mm(a, b)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            flops = 2 * size**3  # FLOPs for matrix multiplication
            gflops = (flops / avg_time) / 1e9
            
            results[f'matrix_{size}x{size}'] = {
                'avg_time': avg_time,
                'gflops': gflops,
                'matrix_size': size
            }
        
        return results
    
    def benchmark_tensor_operations(self):
        """Benchmark various tensor operations"""
        print("Running tensor operation benchmarks...")
        
        results = {}
        
        # Test different tensor operations
        operations = {
            'conv2d': self._benchmark_conv2d,
            'maxpool2d': self._benchmark_maxpool2d,
            'relu': self._benchmark_relu,
            'batch_norm': self._benchmark_batch_norm
        }
        
        for op_name, op_func in operations.items():
            print(f"  Testing {op_name}...")
            results[op_name] = op_func()
        
        return results
    
    def _benchmark_conv2d(self):
        """Benchmark 2D convolution"""
        conv = torch.nn.Conv2d(64, 128, 3, padding=1)
        input_tensor = torch.randn(4, 64, 224, 224)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            output = conv(input_tensor)
            times.append(time.time() - start_time)
        
        return {'avg_time': np.mean(times), 'std_time': np.std(times)}
    
    def _benchmark_maxpool2d(self):
        """Benchmark 2D max pooling"""
        pool = torch.nn.MaxPool2d(2, 2)
        input_tensor = torch.randn(4, 128, 224, 224)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            output = pool(input_tensor)
            times.append(time.time() - start_time)
        
        return {'avg_time': np.mean(times), 'std_time': np.std(times)}
    
    def _benchmark_relu(self):
        """Benchmark ReLU activation"""
        relu = torch.nn.ReLU()
        input_tensor = torch.randn(4, 128, 224, 224)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            output = relu(input_tensor)
            times.append(time.time() - start_time)
        
        return {'avg_time': np.mean(times), 'std_time': np.std(times)}
    
    def _benchmark_batch_norm(self):
        """Benchmark batch normalization"""
        bn = torch.nn.BatchNorm2d(128)
        input_tensor = torch.randn(4, 128, 224, 224)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            output = bn(input_tensor)
            times.append(time.time() - start_time)
        
        return {'avg_time': np.mean(times), 'std_time': np.std(times)}
    
    def monitor_system_resources(self):
        """Monitor system resources during benchmark"""
        print("Monitoring system resources...")
        
        cpu_usage = []
        memory_usage = []
        timestamps = []
        
        start_time = time.time()
        duration = 20  # Monitor for 20 seconds
        
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=0.5))
            memory_usage.append(psutil.virtual_memory().percent)
            timestamps.append(time.time() - start_time)
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamps': timestamps,
            'avg_cpu': np.mean(cpu_usage),
            'avg_memory': np.mean(memory_usage),
            'max_cpu': np.max(cpu_usage),
            'max_memory': np.max(memory_usage)
        }
    
    def run_benchmark(self):
        """Run all benchmarks"""
        print("Starting NVIDIA Jetson Orin Nano ML Benchmark...")
        print(f"System Info: {self.system_info}")
        
        self.results['system_info'] = self.system_info
        self.results['pytorch_models'] = self.benchmark_pytorch_models()
        self.results['matrix_operations'] = self.benchmark_matrix_operations()
        self.results['tensor_operations'] = self.benchmark_tensor_operations()
        self.results['system_monitoring'] = self.monitor_system_resources()
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.results['total_benchmark_time'] = total_time
        
        return self.results
    
    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "="*60)
        print("NVIDIA JETSON ORIN NANO ML BENCHMARK REPORT")
        print("="*60)
        
        print(f"\nSystem Information:")
        print(f"  Model: Jetson Orin Nano")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Total Memory: {self.system_info['memory_total']:.1f} GB")
        print(f"  Available Memory: {self.system_info['memory_available']:.1f} GB")
        print(f"  CUDA Available: {self.system_info['cuda_available']}")
        print(f"  PyTorch Version: {self.system_info['pytorch_version']}")
        
        print(f"\nPyTorch Model Performance (CPU):")
        for model, metrics in self.results['pytorch_models'].items():
            print(f"  {model}:")
            print(f"    Inference Time: {metrics['avg_inference_time']*1000:.2f} ± {metrics['std_inference_time']*1000:.2f} ms")
            print(f"    Throughput: {metrics['throughput_fps']:.2f} FPS")
            print(f"    Model Parameters: {metrics['model_params']:,}")
            print(f"    Model Size: {metrics['model_size_mb']:.2f} MB")
        
        print(f"\nMatrix Operation Performance:")
        for operation, metrics in self.results['matrix_operations'].items():
            print(f"  {operation}:")
            print(f"    Time: {metrics['avg_time']*1000:.2f} ms")
            print(f"    Performance: {metrics['gflops']:.2f} GFLOPS")
        
        print(f"\nTensor Operation Performance:")
        for operation, metrics in self.results['tensor_operations'].items():
            print(f"  {operation}:")
            print(f"    Time: {metrics['avg_time']*1000:.2f} ± {metrics['std_time']*1000:.2f} ms")
        
        print(f"\nSystem Resource Usage:")
        monitoring = self.results['system_monitoring']
        print(f"  Average CPU Usage: {monitoring['avg_cpu']:.1f}%")
        print(f"  Peak CPU Usage: {monitoring['max_cpu']:.1f}%")
        print(f"  Average Memory Usage: {monitoring['avg_memory']:.1f}%")
        print(f"  Peak Memory Usage: {monitoring['max_memory']:.1f}%")
        
        print(f"\nTotal Benchmark Time: {self.results['total_benchmark_time']:.1f} seconds")
        
        # Save results to JSON
        with open('gb10_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: gb10_benchmark_results.json")

if __name__ == "__main__":
    benchmark = SimpleJetsonBenchmark()
    results = benchmark.run_benchmark()
    benchmark.generate_report()
    
    print("\nBenchmark completed successfully!")

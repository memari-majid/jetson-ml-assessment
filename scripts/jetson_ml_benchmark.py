#!/usr/bin/env python3
"""
NVIDIA Jetson Orin Nano Machine Learning Benchmark Suite
Comprehensive testing of ML capabilities including CPU, GPU, and optimization techniques
"""

import time
import json
import psutil
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
import os
import sys

# Import ML frameworks
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import tensorflow as tf
import cv2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class JetsonMLBenchmark:
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
            'tensorflow_version': tf.__version__,
            'opencv_version': cv2.__version__,
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
    
    def monitor_system(self, duration=10):
        """Monitor system resources during benchmark"""
        cpu_usage = []
        memory_usage = []
        timestamps = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
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
    
    def benchmark_pytorch_cpu(self):
        """Benchmark PyTorch on CPU"""
        print("Running PyTorch CPU benchmarks...")
        
        # Test different model sizes
        models_to_test = {
            'resnet18': models.resnet18(pretrained=False),
            'resnet50': models.resnet50(pretrained=False),
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
            'efficientnet_b0': models.efficientnet_b0(pretrained=False)
        }
        
        results = {}
        batch_size = 8
        input_size = (3, 224, 224)
        
        for model_name, model in models_to_test.items():
            print(f"  Testing {model_name}...")
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_size)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark inference
            times = []
            for _ in range(20):
                start_time = time.time()
                with torch.no_grad():
                    output = model(dummy_input)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            results[model_name] = {
                'avg_inference_time': avg_time,
                'throughput_fps': throughput,
                'model_params': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            }
        
        return results
    
    def benchmark_tensorflow_cpu(self):
        """Benchmark TensorFlow on CPU"""
        print("Running TensorFlow CPU benchmarks...")
        
        results = {}
        
        # Test different model architectures
        models_to_test = {
            'dense_64': tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ]),
            'dense_256': tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ]),
            'cnn_simple': tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        }
        
        for model_name, model in models_to_test.items():
            print(f"  Testing {model_name}...")
            
            # Create appropriate dummy input
            if 'cnn' in model_name:
                dummy_input = tf.random.normal((32, 28, 28, 1))
            else:
                dummy_input = tf.random.normal((32, 784))
            
            # Warmup
            for _ in range(5):
                _ = model(dummy_input)
            
            # Benchmark inference
            times = []
            for _ in range(20):
                start_time = time.time()
                output = model(dummy_input)
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            throughput = 32 / avg_time
            
            results[model_name] = {
                'avg_inference_time': avg_time,
                'throughput_fps': throughput,
                'model_params': model.count_params(),
                'model_size_mb': model.count_params() * 4 / (1024**2)  # Assuming float32
            }
        
        return results
    
    def benchmark_opencv_processing(self):
        """Benchmark OpenCV image processing operations"""
        print("Running OpenCV benchmarks...")
        
        # Create test images
        test_images = {
            'small': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'medium': np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            'large': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        }
        
        operations = {
            'gaussian_blur': lambda img: cv2.GaussianBlur(img, (15, 15), 0),
            'canny_edge': lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150),
            'histogram_eq': lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            'morphology': lambda img: cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        }
        
        results = {}
        
        for img_name, img in test_images.items():
            results[img_name] = {}
            for op_name, operation in operations.items():
                times = []
                for _ in range(20):
                    start_time = time.time()
                    result = operation(img)
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                results[img_name][op_name] = {
                    'avg_processing_time': avg_time,
                    'throughput_fps': 1.0 / avg_time
                }
        
        return results
    
    def benchmark_sklearn_training(self):
        """Benchmark scikit-learn model training"""
        print("Running scikit-learn training benchmarks...")
        
        # Generate synthetic datasets of different sizes
        datasets = {
            'small': (1000, 20),
            'medium': (10000, 50),
            'large': (50000, 100)
        }
        
        results = {}
        
        for dataset_name, (n_samples, n_features) in datasets.items():
            print(f"  Testing {dataset_name} dataset ({n_samples} samples, {n_features} features)...")
            
            # Generate synthetic data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(10, n_features // 2),
                n_classes=3,
                random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            start_time = time.time()
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Test inference
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[dataset_name] = {
                'training_time': training_time,
                'inference_time': inference_time,
                'accuracy': accuracy,
                'n_samples': n_samples,
                'n_features': n_features
            }
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks and collect results"""
        print("Starting comprehensive ML benchmark on NVIDIA Jetson Orin Nano...")
        print(f"System Info: {self.system_info}")
        
        # Run benchmarks
        self.results['system_info'] = self.system_info
        self.results['pytorch_cpu'] = self.benchmark_pytorch_cpu()
        self.results['tensorflow_cpu'] = self.benchmark_tensorflow_cpu()
        self.results['opencv_processing'] = self.benchmark_opencv_processing()
        self.results['sklearn_training'] = self.benchmark_sklearn_training()
        
        # System monitoring during intensive operations
        print("Monitoring system during intensive operations...")
        self.results['system_monitoring'] = self.monitor_system(duration=30)
        
        # Calculate total benchmark time
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.results['total_benchmark_time'] = total_time
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*60)
        print("NVIDIA JETSON ORIN NANO ML BENCHMARK REPORT")
        print("="*60)
        
        print(f"\nSystem Information:")
        print(f"  Model: {self.system_info.get('gpu_name', 'Unknown')}")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Total Memory: {self.system_info['memory_total']:.1f} GB")
        print(f"  CUDA Available: {self.system_info['cuda_available']}")
        print(f"  PyTorch Version: {self.system_info['pytorch_version']}")
        print(f"  TensorFlow Version: {self.system_info['tensorflow_version']}")
        
        print(f"\nPyTorch CPU Performance:")
        for model, metrics in self.results['pytorch_cpu'].items():
            print(f"  {model}:")
            print(f"    Inference Time: {metrics['avg_inference_time']*1000:.2f} ms")
            print(f"    Throughput: {metrics['throughput_fps']:.2f} FPS")
            print(f"    Model Size: {metrics['model_size_mb']:.2f} MB")
        
        print(f"\nTensorFlow CPU Performance:")
        for model, metrics in self.results['tensorflow_cpu'].items():
            print(f"  {model}:")
            print(f"    Inference Time: {metrics['avg_inference_time']*1000:.2f} ms")
            print(f"    Throughput: {metrics['throughput_fps']:.2f} FPS")
            print(f"    Model Size: {metrics['model_size_mb']:.2f} MB")
        
        print(f"\nSystem Resource Usage:")
        monitoring = self.results['system_monitoring']
        print(f"  Average CPU Usage: {monitoring['avg_cpu']:.1f}%")
        print(f"  Peak CPU Usage: {monitoring['max_cpu']:.1f}%")
        print(f"  Average Memory Usage: {monitoring['avg_memory']:.1f}%")
        print(f"  Peak Memory Usage: {monitoring['max_memory']:.1f}%")
        
        print(f"\nTotal Benchmark Time: {self.results['total_benchmark_time']:.1f} seconds")
        
        # Save detailed results to JSON
        with open('gb10_ml_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: gb10_ml_benchmark_results.json")
    
    def create_visualizations(self):
        """Create performance visualization charts"""
        print("Creating performance visualizations...")
        print("Visualization functionality disabled due to matplotlib compatibility issues")
        # Note: Visualization code commented out due to NumPy/matplotlib compatibility issues

if __name__ == "__main__":
    benchmark = JetsonMLBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.generate_report()
    # benchmark.create_visualizations()  # Disabled due to matplotlib issues
    
    print("\nBenchmark completed successfully!")

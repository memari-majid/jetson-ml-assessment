#!/usr/bin/env python3
"""
TensorRT Model Optimization Script
Converts PyTorch models to TensorRT optimized engines
"""

import torch
import torchvision.models as models
import time
import json
import sys
import os

class TensorRTOptimizer:
    def __init__(self):
        self.available = self.check_tensorrt()
        
    def check_tensorrt(self):
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            print(f"✅ TensorRT version: {trt.__version__}")
            return True
        except ImportError:
            print("❌ TensorRT not installed")
            print("   Install with: pip3 install nvidia-tensorrt")
            print("   Or see NEXT_STEPS_PLAN.md for full installation")
            return False
    
    def check_torch_tensorrt(self):
        """Check if torch-tensorrt is available"""
        try:
            import torch_tensorrt
            print(f"✅ torch-tensorrt available")
            return True
        except ImportError:
            print("⚠️  torch-tensorrt not installed")
            print("   Install with: pip3 install torch-tensorrt")
            return False
    
    def optimize_model_fp16(self, model, model_name, batch_size=4):
        """
        Optimize model with FP16 precision
        This is a simplified version - full TensorRT integration requires torch-tensorrt
        """
        print(f"\n{'='*60}")
        print(f"  Optimizing {model_name} with FP16")
        print('='*60)
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available - GPU required for TensorRT")
            return None
        
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()
        
        # Convert to FP16
        model_fp16 = model.half()
        
        # Create example input
        dummy_input = torch.randn(batch_size, 3, 224, 224).half().to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_fp16(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                output = model_fp16(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        import numpy as np
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        
        print(f"✅ FP16 Performance:")
        print(f"   Throughput: {throughput:.2f} FPS")
        print(f"   Latency: {avg_time*1000/batch_size:.2f} ms per image")
        
        return {
            'model_name': model_name,
            'precision': 'fp16',
            'batch_size': batch_size,
            'throughput_fps': throughput,
            'avg_latency_ms': avg_time * 1000 / batch_size
        }
    
    def optimize_with_torch_tensorrt(self, model, model_name, batch_size=4):
        """
        Optimize model using torch-tensorrt (if available)
        """
        if not self.check_torch_tensorrt():
            print("⚠️  Skipping torch-tensorrt optimization")
            return None
        
        print(f"\n{'='*60}")
        print(f"  Optimizing {model_name} with TensorRT")
        print('='*60)
        
        try:
            import torch_tensorrt
            
            device = torch.device('cuda')
            model = model.to(device)
            model.eval()
            
            # Create example input
            example_input = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Compile with TensorRT
            print("   Compiling model (this may take a few minutes)...")
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30  # 1GB
            )
            
            # Benchmark
            torch.cuda.synchronize()
            times = []
            for _ in range(50):
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    output = trt_model(example_input)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            import numpy as np
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            print(f"✅ TensorRT Performance:")
            print(f"   Throughput: {throughput:.2f} FPS")
            print(f"   Latency: {avg_time*1000/batch_size:.2f} ms per image")
            
            # Save optimized model
            output_file = f"{model_name}_tensorrt.ts"
            torch.jit.save(trt_model, output_file)
            print(f"   Saved to: {output_file}")
            
            return {
                'model_name': model_name,
                'precision': 'fp16_tensorrt',
                'batch_size': batch_size,
                'throughput_fps': throughput,
                'avg_latency_ms': avg_time * 1000 / batch_size,
                'saved_file': output_file
            }
            
        except Exception as e:
            print(f"❌ Error during TensorRT optimization: {e}")
            return None
    
    def benchmark_quantization(self, model, model_name, batch_size=4):
        """
        Benchmark INT8 quantization (simplified version)
        """
        print(f"\n{'='*60}")
        print(f"  Testing INT8 Quantization for {model_name}")
        print('='*60)
        
        if not torch.cuda.is_available():
            print("⚠️  Quantization works best on GPU, using CPU fallback")
        
        try:
            # This is a simplified dynamic quantization example
            # Full INT8 quantization with calibration is more complex
            model_quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            model_quantized.eval()
            
            # Create input
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            # Benchmark
            times = []
            for _ in range(30):
                start = time.time()
                with torch.no_grad():
                    output = model_quantized(dummy_input)
                times.append(time.time() - start)
            
            import numpy as np
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            print(f"✅ INT8 Quantized Performance:")
            print(f"   Throughput: {throughput:.2f} FPS")
            print(f"   Latency: {avg_time*1000/batch_size:.2f} ms per image")
            
            # Model size comparison
            import tempfile
            
            # Original model size
            with tempfile.NamedTemporaryFile() as f:
                torch.save(model.state_dict(), f.name)
                original_size = os.path.getsize(f.name) / (1024**2)
            
            # Quantized model size
            with tempfile.NamedTemporaryFile() as f:
                torch.save(model_quantized.state_dict(), f.name)
                quantized_size = os.path.getsize(f.name) / (1024**2)
            
            print(f"   Model size reduction: {original_size:.2f} MB → {quantized_size:.2f} MB")
            print(f"   Compression ratio: {original_size/quantized_size:.2f}x")
            
            return {
                'model_name': model_name,
                'precision': 'int8',
                'batch_size': batch_size,
                'throughput_fps': throughput,
                'avg_latency_ms': avg_time * 1000 / batch_size,
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size
            }
            
        except Exception as e:
            print(f"❌ Error during quantization: {e}")
            return None
    
    def run_optimization_suite(self):
        """Run optimization benchmarks on standard models"""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  TensorRT and Optimization Benchmark Suite                  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        if not self.available:
            print("\n❌ TensorRT not available. Install required packages:")
            print("   pip3 install nvidia-tensorrt torch-tensorrt")
            sys.exit(1)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimizations': []
        }
        
        models_to_test = {
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
            'resnet18': models.resnet18(pretrained=False),
        }
        
        for model_name, model in models_to_test.items():
            print(f"\n{'#'*60}")
            print(f"  Processing: {model_name}")
            print('#'*60)
            
            # FP16 optimization
            result_fp16 = self.optimize_model_fp16(model, model_name)
            if result_fp16:
                results['optimizations'].append(result_fp16)
            
            # TensorRT optimization
            result_trt = self.optimize_with_torch_tensorrt(model, model_name)
            if result_trt:
                results['optimizations'].append(result_trt)
            
            # Quantization
            result_quant = self.benchmark_quantization(model, model_name)
            if result_quant:
                results['optimizations'].append(result_quant)
        
        # Save results
        output_file = 'tensorrt_optimization_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print('='*60)
        
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print optimization summary"""
        print(f"\n{'='*60}")
        print("  Optimization Summary")
        print('='*60)
        
        if not results['optimizations']:
            print("No optimizations completed")
            return
        
        print(f"\n{'Model':<15} {'Precision':<15} {'FPS':<12} {'Latency (ms)':<15}")
        print("-" * 60)
        
        for opt in results['optimizations']:
            model = opt.get('model_name', 'unknown')
            precision = opt.get('precision', 'unknown')
            fps = opt.get('throughput_fps', 0)
            latency = opt.get('avg_latency_ms', 0)
            
            print(f"{model:<15} {precision:<15} {fps:>10.2f}  {latency:>13.2f}")
        
        print("\n" + "="*60)

def main():
    optimizer = TensorRTOptimizer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print("Checking TensorRT availability...")
        optimizer.check_torch_tensorrt()
        sys.exit(0)
    
    optimizer.run_optimization_suite()

if __name__ == "__main__":
    main()


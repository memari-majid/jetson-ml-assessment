#!/usr/bin/env python3
"""
Compare benchmark results between CPU and GPU, or different runs
Generates comparison reports and improvement metrics
"""

import json
import sys
from datetime import datetime

class BenchmarkComparator:
    def __init__(self, cpu_file, gpu_file=None):
        self.cpu_file = cpu_file
        self.gpu_file = gpu_file
        self.cpu_data = None
        self.gpu_data = None
        
    def load_results(self):
        """Load benchmark results from JSON files"""
        try:
            with open(self.cpu_file, 'r') as f:
                self.cpu_data = json.load(f)
            print(f"✅ Loaded CPU results from: {self.cpu_file}")
        except FileNotFoundError:
            print(f"❌ CPU results file not found: {self.cpu_file}")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in CPU results file")
            return False
        
        if self.gpu_file:
            try:
                with open(self.gpu_file, 'r') as f:
                    self.gpu_data = json.load(f)
                print(f"✅ Loaded GPU results from: {self.gpu_file}")
            except FileNotFoundError:
                print(f"⚠️  GPU results file not found: {self.gpu_file}")
                print("   Continuing with CPU-only analysis")
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON in GPU results file")
        
        return True
    
    def compare_model_performance(self):
        """Compare model inference performance"""
        print("\n" + "="*70)
        print("  Model Performance Comparison")
        print("="*70)
        
        if not self.cpu_data:
            return
        
        cpu_models = self.cpu_data.get('pytorch_models', {})
        
        if self.gpu_data:
            gpu_models = self.gpu_data.get('pytorch_gpu', {})
            
            print(f"\n{'Model':<15} {'CPU FPS':<12} {'GPU FPS':<12} {'Speedup':<10}")
            print("-" * 70)
            
            for model_name in cpu_models.keys():
                cpu_fps = cpu_models[model_name].get('throughput_fps', 0)
                
                if model_name in gpu_models and 'batch_4' in gpu_models[model_name]:
                    gpu_fps = gpu_models[model_name]['batch_4'].get('throughput_fps', 0)
                    speedup = gpu_fps / cpu_fps if cpu_fps > 0 else 0
                    
                    print(f"{model_name:<15} {cpu_fps:>10.2f}  {gpu_fps:>10.2f}  {speedup:>8.2f}x")
                else:
                    print(f"{model_name:<15} {cpu_fps:>10.2f}  {'N/A':<12} {'N/A':<10}")
        else:
            print(f"\n{'Model':<15} {'FPS':<12} {'Latency (ms)':<15}")
            print("-" * 70)
            
            for model_name, metrics in cpu_models.items():
                fps = metrics.get('throughput_fps', 0)
                latency = metrics.get('avg_inference_time', 0) * 1000
                print(f"{model_name:<15} {fps:>10.2f}  {latency:>13.2f}")
    
    def compare_compute_performance(self):
        """Compare matrix operation performance"""
        print("\n" + "="*70)
        print("  Compute Performance Comparison")
        print("="*70)
        
        if not self.cpu_data:
            return
        
        cpu_ops = self.cpu_data.get('matrix_operations', {})
        
        if self.gpu_data:
            gpu_ops = self.gpu_data.get('cuda_operations', {})
            
            print(f"\n{'Operation':<20} {'CPU GFLOPS':<15} {'GPU GFLOPS':<15} {'Speedup':<10}")
            print("-" * 70)
            
            for op_name, cpu_metrics in cpu_ops.items():
                cpu_gflops = cpu_metrics.get('gflops', 0)
                
                # Try to find matching GPU operation
                gpu_op_name = op_name.replace('matrix_', 'matmul_')
                if gpu_op_name in gpu_ops:
                    gpu_gflops = gpu_ops[gpu_op_name].get('gflops', 0)
                    speedup = gpu_gflops / cpu_gflops if cpu_gflops > 0 else 0
                    
                    print(f"{op_name:<20} {cpu_gflops:>13.2f}  {gpu_gflops:>13.2f}  {speedup:>8.2f}x")
                else:
                    print(f"{op_name:<20} {cpu_gflops:>13.2f}  {'N/A':<15} {'N/A':<10}")
        else:
            print(f"\n{'Operation':<20} {'GFLOPS':<15}")
            print("-" * 70)
            
            for op_name, metrics in cpu_ops.items():
                gflops = metrics.get('gflops', 0)
                print(f"{op_name:<20} {gflops:>13.2f}")
    
    def compare_system_resources(self):
        """Compare system resource utilization"""
        print("\n" + "="*70)
        print("  System Resource Utilization")
        print("="*70)
        
        if not self.cpu_data:
            return
        
        cpu_mon = self.cpu_data.get('system_monitoring', {})
        
        print(f"\n{'Metric':<25} {'CPU Run':<15}")
        print("-" * 70)
        print(f"{'Average CPU Usage':<25} {cpu_mon.get('avg_cpu', 0):>13.1f}%")
        print(f"{'Peak CPU Usage':<25} {cpu_mon.get('max_cpu', 0):>13.1f}%")
        print(f"{'Average Memory Usage':<25} {cpu_mon.get('avg_memory', 0):>13.1f}%")
        print(f"{'Peak Memory Usage':<25} {cpu_mon.get('max_memory', 0):>13.1f}%")
        
        if self.gpu_data:
            gpu_mon = self.gpu_data.get('gpu_monitoring', {})
            if gpu_mon:
                print(f"\n{'GPU Metric':<25} {'Value':<15}")
                print("-" * 70)
                print(f"{'Avg GPU Memory (MB)':<25} {gpu_mon.get('avg_memory_mb', 0):>13.2f}")
                print(f"{'Max GPU Memory (MB)':<25} {gpu_mon.get('max_memory_mb', 0):>13.2f}")
    
    def generate_summary(self):
        """Generate overall comparison summary"""
        print("\n" + "="*70)
        print("  Summary and Recommendations")
        print("="*70)
        
        if self.gpu_data:
            print("\n✅ GPU Acceleration Available")
            print("\nKey Findings:")
            
            # Calculate average speedup
            cpu_models = self.cpu_data.get('pytorch_models', {})
            gpu_models = self.gpu_data.get('pytorch_gpu', {})
            
            speedups = []
            for model_name in cpu_models.keys():
                cpu_fps = cpu_models[model_name].get('throughput_fps', 0)
                if model_name in gpu_models and 'batch_4' in gpu_models[model_name]:
                    gpu_fps = gpu_models[model_name]['batch_4'].get('throughput_fps', 0)
                    if cpu_fps > 0:
                        speedups.append(gpu_fps / cpu_fps)
            
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"  - Average GPU Speedup: {avg_speedup:.2f}x")
                print(f"  - Best GPU Speedup: {max(speedups):.2f}x")
                print(f"  - Minimum GPU Speedup: {min(speedups):.2f}x")
            
            print("\nRecommendations:")
            print("  ✅ Use GPU for all inference workloads")
            print("  ✅ Consider TensorRT for further optimization")
            print("  ✅ Explore INT8 quantization for 2-4x additional speedup")
        else:
            print("\n⚠️  GPU Results Not Available")
            print("\nCurrent Performance (CPU-only):")
            
            cpu_models = self.cpu_data.get('pytorch_models', {})
            if cpu_models:
                avg_fps = sum(m.get('throughput_fps', 0) for m in cpu_models.values()) / len(cpu_models)
                print(f"  - Average Throughput: {avg_fps:.2f} FPS")
            
            print("\nRecommendations:")
            print("  🔴 Enable GPU access for 5-10x performance improvement")
            print("  📖 See NEXT_STEPS_PLAN.md for GPU enablement instructions")
            print("  🔧 Install PyTorch with CUDA support")
    
    def run_comparison(self):
        """Run full comparison analysis"""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Jetson Benchmark Results Comparison Tool                   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        if not self.load_results():
            return False
        
        self.compare_model_performance()
        self.compare_compute_performance()
        self.compare_system_resources()
        self.generate_summary()
        
        print("\n" + "="*70)
        print("Comparison complete!")
        print("="*70 + "\n")
        
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_results.py <cpu_results.json> [gpu_results.json]")
        print("\nExamples:")
        print("  python3 compare_results.py jetson_benchmark_results.json")
        print("  python3 compare_results.py jetson_benchmark_results.json jetson_gpu_benchmark_results.json")
        sys.exit(1)
    
    cpu_file = sys.argv[1]
    gpu_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    comparator = BenchmarkComparator(cpu_file, gpu_file)
    success = comparator.run_comparison()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


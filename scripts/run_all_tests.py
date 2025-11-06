#!/usr/bin/env python3
"""
Automated Test Runner for Jetson ML Benchmark Suite
Runs all available tests and generates comprehensive report
"""

import os
import sys
import subprocess
import json
from datetime import datetime
import argparse

class JetsonTestRunner:
    def __init__(self, skip_gpu=False, quick=False):
        self.skip_gpu = skip_gpu
        self.quick = quick
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': [],
            'tests_passed': [],
            'tests_failed': [],
            'test_results': {}
        }
        
    def run_command(self, cmd, timeout=300):
        """Run a shell command and capture output"""
        try:
            print(f"\n{'='*60}")
            print(f"Running: {cmd}")
            print('='*60)
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def run_verification(self):
        """Run system verification"""
        print("\n" + "="*60)
        print("  STEP 1: System Verification")
        print("="*60)
        
        result = self.run_command('cd /home/majid/Downloads/jetson-ml-assessment/scripts && python3 jetson_verify.py', timeout=60)
        
        self.results['tests_run'].append('verification')
        if result['success']:
            self.results['tests_passed'].append('verification')
            print("✅ System verification passed")
        else:
            self.results['tests_failed'].append('verification')
            print("⚠️  System verification found issues")
        
        self.results['test_results']['verification'] = result
        return result['success']
    
    def run_cpu_benchmark(self):
        """Run CPU-only benchmark"""
        print("\n" + "="*60)
        print("  STEP 2: CPU Benchmark")
        print("="*60)
        
        if self.quick:
            print("⏩ Skipping (quick mode enabled)")
            return True
        
        result = self.run_command('cd /home/majid/Downloads/jetson-ml-assessment/scripts && python3 jetson_simple_benchmark.py', timeout=300)
        
        self.results['tests_run'].append('cpu_benchmark')
        if result['success']:
            self.results['tests_passed'].append('cpu_benchmark')
            print("✅ CPU benchmark completed")
            
            # Try to load results
            try:
                with open('../data/jetson_benchmark_results.json', 'r') as f:
                    cpu_results = json.load(f)
                    self.results['test_results']['cpu_benchmark'] = cpu_results
            except:
                pass
        else:
            self.results['tests_failed'].append('cpu_benchmark')
            print("❌ CPU benchmark failed")
        
        return result['success']
    
    def run_gpu_benchmark(self):
        """Run GPU benchmark"""
        print("\n" + "="*60)
        print("  STEP 3: GPU Benchmark")
        print("="*60)
        
        if self.skip_gpu:
            print("⏩ Skipping (GPU tests disabled)")
            return True
        
        # Check if GPU is available first
        check_result = self.run_command(
            'python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"',
            timeout=10
        )
        
        if not check_result['success']:
            print("⚠️  GPU not available, skipping GPU tests")
            print("   See NEXT_STEPS_PLAN.md for GPU enablement")
            return True  # Not a failure, just skip
        
        result = self.run_command('cd /home/majid/Downloads/jetson-ml-assessment/scripts && python3 jetson_gpu_benchmark.py', timeout=300)
        
        self.results['tests_run'].append('gpu_benchmark')
        if result['success']:
            self.results['tests_passed'].append('gpu_benchmark')
            print("✅ GPU benchmark completed")
            
            # Try to load results
            try:
                with open('../data/jetson_gpu_benchmark_results.json', 'r') as f:
                    gpu_results = json.load(f)
                    self.results['test_results']['gpu_benchmark'] = gpu_results
            except:
                pass
        else:
            self.results['tests_failed'].append('gpu_benchmark')
            print("❌ GPU benchmark failed")
        
        return result['success']
    
    def generate_summary_report(self):
        """Generate final summary report"""
        print("\n" + "="*70)
        print("  JETSON ML BENCHMARK SUITE - FINAL REPORT")
        print("="*70)
        
        print(f"\nTest Execution Summary:")
        print(f"  Total Tests Run: {len(self.results['tests_run'])}")
        print(f"  Tests Passed: {len(self.results['tests_passed'])}")
        print(f"  Tests Failed: {len(self.results['tests_failed'])}")
        
        if self.results['tests_passed']:
            print(f"\n✅ Passed Tests:")
            for test in self.results['tests_passed']:
                print(f"   - {test}")
        
        if self.results['tests_failed']:
            print(f"\n❌ Failed Tests:")
            for test in self.results['tests_failed']:
                print(f"   - {test}")
        
        # Performance summary
        print(f"\n{'='*70}")
        print("  Performance Summary")
        print('='*70)
        
        if 'cpu_benchmark' in self.results['test_results']:
            cpu_data = self.results['test_results']['cpu_benchmark']
            if 'pytorch_models' in cpu_data:
                print("\nCPU Performance:")
                for model, metrics in cpu_data['pytorch_models'].items():
                    fps = metrics.get('throughput_fps', 0)
                    print(f"  {model}: {fps:.2f} FPS")
        
        if 'gpu_benchmark' in self.results['test_results']:
            gpu_data = self.results['test_results']['gpu_benchmark']
            if 'pytorch_gpu' in gpu_data:
                print("\nGPU Performance:")
                for model, metrics in gpu_data['pytorch_gpu'].items():
                    if 'batch_8' in metrics:
                        fps = metrics['batch_8'].get('throughput_fps', 0)
                        print(f"  {model}: {fps:.2f} FPS (batch 8)")
        
        # Save comprehensive results
        output_file = '../data/jetson_test_suite_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"Comprehensive results saved to: {output_file}")
        print('='*70)
        
        # Final status
        success = len(self.results['tests_failed']) == 0
        if success:
            print("\n✅ All tests completed successfully!")
        else:
            print("\n⚠️  Some tests failed. Review the output above.")
        
        return success
    
    def run_all(self):
        """Run all tests in sequence"""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  NVIDIA Jetson ML Benchmark Suite - Automated Test Runner   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        # Run tests
        self.run_verification()
        self.run_cpu_benchmark()
        self.run_gpu_benchmark()
        
        # Generate report
        success = self.generate_summary_report()
        
        return 0 if success else 1

def main():
    parser = argparse.ArgumentParser(
        description='Automated test runner for Jetson ML benchmarks'
    )
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Skip GPU benchmark tests'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode - skip long-running tests'
    )
    
    args = parser.parse_args()
    
    runner = JetsonTestRunner(skip_gpu=args.skip_gpu, quick=args.quick)
    exit_code = runner.run_all()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()


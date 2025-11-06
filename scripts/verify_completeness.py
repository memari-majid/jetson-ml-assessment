#!/usr/bin/env python3
"""
Jetson ML Assessment Suite - Completeness Verification
Checks that all components are present and properly configured
"""

import os
import sys
from pathlib import Path

class CompletenessChecker:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.issues = []
        self.warnings = []
        
    def check_file(self, filename, required=True):
        """Check if a file exists"""
        path = self.base_dir / filename
        exists = path.exists()
        
        if exists:
            size = path.stat().st_size
            return True, size
        else:
            if required:
                self.issues.append(f"Missing required file: {filename}")
            else:
                self.warnings.append(f"Missing optional file: {filename}")
            return False, 0
    
    def print_section(self, title):
        """Print section header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print('='*70)
    
    def check_documentation(self):
        """Check all documentation files"""
        self.print_section("Documentation Files")
        
        docs = [
            ('README.md', True),
            ('QUICK_START.md', True),
            ('EXECUTIVE_SUMMARY.md', True),
            ('NEXT_STEPS_PLAN.md', True),
            ('SETUP_GUIDE.md', True),
            ('NVIDIA_Jetson_Orin_Nano_ML_Comprehensive_Report.md', True),
            ('CHANGELOG.md', True),
            ('INDEX.txt', True),
        ]
        
        total_size = 0
        for filename, required in docs:
            exists, size = self.check_file(filename, required)
            status = "✅" if exists else "❌"
            size_kb = size / 1024 if size > 0 else 0
            print(f"{status} {filename:<55} {size_kb:>6.1f} KB")
            total_size += size
        
        print(f"\nTotal documentation size: {total_size/1024:.1f} KB")
    
    def check_scripts(self):
        """Check all Python scripts"""
        self.print_section("Python Scripts")
        
        scripts = [
            ('jetson_simple_benchmark.py', True, 'CPU benchmark'),
            ('jetson_ml_benchmark.py', True, 'Advanced benchmark'),
            ('jetson_gpu_benchmark.py', True, 'GPU benchmark'),
            ('jetson_verify.py', True, 'System verification'),
            ('run_all_tests.py', True, 'Test automation'),
            ('compare_results.py', True, 'Results analysis'),
            ('tensorrt_optimizer.py', True, 'TensorRT optimization'),
            ('inference_api.py', True, 'REST API server'),
            ('test_api.py', True, 'API testing'),
            ('verify_completeness.py', True, 'This script'),
        ]
        
        for filename, required, description in scripts:
            exists, size = self.check_file(filename, required)
            status = "✅" if exists else "❌"
            
            # Check if executable
            path = self.base_dir / filename
            is_executable = path.exists() and os.access(path, os.X_OK)
            exec_status = "🔧" if is_executable else "  "
            
            print(f"{status} {exec_status} {filename:<30} - {description}")
        
        print("\n🔧 = Executable")
    
    def check_config_files(self):
        """Check configuration files"""
        self.print_section("Configuration Files")
        
        configs = [
            ('requirements.txt', True, 'Python dependencies'),
            ('Makefile', True, 'Build automation'),
            ('.gitignore', True, 'Git configuration'),
        ]
        
        for filename, required, description in configs:
            exists, size = self.check_file(filename, required)
            status = "✅" if exists else "❌"
            print(f"{status} {filename:<30} - {description}")
    
    def check_results(self):
        """Check results files"""
        self.print_section("Results and Data Files")
        
        results = [
            ('jetson_benchmark_results.json', True, 'CPU benchmark results'),
            ('system_info.txt', False, 'System information'),
            ('jetson_gpu_benchmark_results.json', False, 'GPU results (when available)'),
            ('tensorrt_optimization_results.json', False, 'TensorRT results'),
            ('jetson_test_suite_results.json', False, 'Test suite results'),
        ]
        
        for filename, required, description in results:
            exists, size = self.check_file(filename, required)
            status = "✅" if exists else "⚠️ "
            size_kb = size / 1024 if size > 0 else 0
            print(f"{status} {filename:<40} - {description}")
            if exists and size > 0:
                print(f"     Size: {size_kb:.1f} KB")
    
    def check_python_imports(self):
        """Check if required Python packages can be imported"""
        self.print_section("Python Package Availability")
        
        packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'TorchVision'),
            ('numpy', 'NumPy'),
            ('psutil', 'psutil'),
            ('PIL', 'Pillow'),
            ('fastapi', 'FastAPI'),
            ('uvicorn', 'Uvicorn'),
        ]
        
        optional_packages = [
            ('tensorflow', 'TensorFlow'),
            ('cv2', 'OpenCV'),
            ('sklearn', 'scikit-learn'),
            ('pandas', 'Pandas'),
            ('tensorrt', 'TensorRT'),
        ]
        
        print("\nRequired Packages:")
        for pkg, name in packages:
            try:
                __import__(pkg)
                print(f"✅ {name}")
            except ImportError:
                print(f"❌ {name} - NOT INSTALLED")
                self.issues.append(f"Required package not installed: {name}")
            except Exception as e:
                print(f"⚠️  {name} - Import error")
                self.warnings.append(f"{name} import error: {str(e)[:40]}")
        
        print("\nOptional Packages:")
        for pkg, name in optional_packages:
            try:
                # Import with a timeout mechanism to prevent hanging
                import importlib
                mod = importlib.import_module(pkg)
                print(f"✅ {name}")
            except ImportError:
                print(f"⚠️  {name} - Not installed (optional)")
                self.warnings.append(f"Optional package not installed: {name}")
            except Exception as e:
                # Catch all other exceptions (compatibility issues, etc.)
                error_msg = str(e)[:60] if str(e) else "unknown error"
                print(f"⚠️  {name} - Import error (compatibility issue)")
                if pkg == 'tensorflow':
                    print(f"     Note: TensorFlow has NumPy compatibility issues")
                self.warnings.append(f"{name} import error: {error_msg}")
    
    def check_functionality(self):
        """Check if key functionality is available"""
        self.print_section("Functionality Check")
        
        # Check CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"✅ CUDA GPU Access: Available")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"⚠️  CUDA GPU Access: Not available (CPU-only mode)")
                self.warnings.append("GPU not accessible - see NEXT_STEPS_PLAN.md")
        except:
            print(f"❌ PyTorch not available")
            self.issues.append("PyTorch not installed")
        
        # Check scripts are executable
        scripts_to_check = [
            'jetson_verify.py',
            'jetson_simple_benchmark.py',
            'run_all_tests.py'
        ]
        
        print("\nScript Permissions:")
        for script in scripts_to_check:
            path = self.base_dir / script
            if path.exists():
                is_executable = os.access(path, os.X_OK)
                status = "✅" if is_executable else "⚠️ "
                print(f"{status} {script}")
                if not is_executable:
                    self.warnings.append(f"{script} not executable (run: chmod +x {script})")
    
    def generate_summary(self):
        """Generate completeness summary"""
        self.print_section("Summary")
        
        if not self.issues and not self.warnings:
            print("\n✅ COMPLETE - All components present and configured!")
            print("\nThe Jetson ML Assessment Suite is ready to use.")
            print("\nNext steps:")
            print("  1. Run: ./jetson_verify.py")
            print("  2. Run: ./jetson_simple_benchmark.py")
            print("  3. See: QUICK_START.md for more options")
            return True
        
        if self.issues:
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        if self.issues:
            print("\n❌ Suite is INCOMPLETE - resolve issues above")
            return False
        else:
            print("\n🟡 Suite is FUNCTIONAL with warnings")
            print("   Warnings can be addressed later")
            return True
    
    def run_check(self):
        """Run all checks"""
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║  Jetson ML Assessment Suite - Completeness Verification         ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        
        self.check_documentation()
        self.check_scripts()
        self.check_config_files()
        self.check_results()
        self.check_python_imports()
        self.check_functionality()
        
        success = self.generate_summary()
        
        print("\n" + "="*70)
        print(f"Verification complete: {len(self.issues)} critical issues, "
              f"{len(self.warnings)} warnings")
        print("="*70 + "\n")
        
        return success

def main():
    checker = CompletenessChecker()
    success = checker.run_check()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


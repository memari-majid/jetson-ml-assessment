# Makefile for Jetson ML Benchmark Suite
# Provides convenient commands for common tasks

.PHONY: help verify install test test-cpu test-gpu test-all compare clean

# Default target
help:
	@echo "Jetson ML Benchmark Suite - Available Commands"
	@echo "==============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make verify       - Verify system configuration"
	@echo ""
	@echo "Testing:"
	@echo "  make test-cpu     - Run CPU benchmark"
	@echo "  make test-gpu     - Run GPU benchmark"
	@echo "  make test-all     - Run all tests"
	@echo "  make test-quick   - Run quick tests"
	@echo ""
	@echo "Analysis:"
	@echo "  make compare      - Compare benchmark results"
	@echo "  make optimize     - Run TensorRT optimization"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean up temporary files"
	@echo "  make report       - Show latest results"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip3 install -r requirements.txt
	@echo "✅ Installation complete"

# Verify system
verify:
	@echo "Verifying system configuration..."
	python3 jetson_verify.py

# Run CPU benchmark
test-cpu:
	@echo "Running CPU benchmark..."
	python3 jetson_simple_benchmark.py

# Run GPU benchmark
test-gpu:
	@echo "Running GPU benchmark..."
	python3 jetson_gpu_benchmark.py

# Run all tests
test-all:
	@echo "Running complete test suite..."
	python3 run_all_tests.py

# Quick test mode
test-quick:
	@echo "Running quick tests..."
	python3 run_all_tests.py --quick

# Compare results
compare:
	@echo "Comparing benchmark results..."
	@if [ -f jetson_gpu_benchmark_results.json ]; then \
		python3 compare_results.py jetson_benchmark_results.json jetson_gpu_benchmark_results.json; \
	else \
		python3 compare_results.py jetson_benchmark_results.json; \
	fi

# Run TensorRT optimization
optimize:
	@echo "Running TensorRT optimization..."
	python3 tensorrt_optimizer.py

# Show latest results
report:
	@echo "Latest Benchmark Results:"
	@echo "========================="
	@if [ -f jetson_benchmark_results.json ]; then \
		echo "\nCPU Benchmark:"; \
		cat jetson_benchmark_results.json | python3 -m json.tool | grep -A 2 "throughput_fps"; \
	fi
	@if [ -f jetson_gpu_benchmark_results.json ]; then \
		echo "\nGPU Benchmark:"; \
		cat jetson_gpu_benchmark_results.json | python3 -m json.tool | grep -A 2 "throughput_fps" | head -20; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "*.tmp" -delete
	find . -type f -name "*.swp" -delete
	@echo "✅ Cleanup complete"

# Full clean (including results)
clean-all: clean
	@echo "Removing all results files..."
	rm -f *_results.json
	rm -f *.ts
	rm -f *.engine
	@echo "✅ Full cleanup complete"

# Set executable permissions on scripts
permissions:
	@echo "Setting executable permissions..."
	chmod +x *.py
	@echo "✅ Permissions updated"

# Check for updates
check-updates:
	@echo "Checking for package updates..."
	pip3 list --outdated

# Run tests with different power modes
test-power-modes:
	@echo "Testing different power modes..."
	@echo "Max Performance Mode:"
	sudo nvpmodel -m 0
	sudo jetson_clocks
	python3 jetson_simple_benchmark.py
	@echo "15W Mode:"
	sudo nvpmodel -m 1
	python3 jetson_simple_benchmark.py

# Complete workflow
all: verify test-cpu compare
	@echo ""
	@echo "✅ Complete benchmark workflow finished"
	@echo "   Check jetson_benchmark_results.json for details"

# Development mode - run with file watching
dev:
	@echo "Development mode - watching for changes..."
	@echo "Press Ctrl+C to stop"
	while true; do \
		python3 jetson_verify.py; \
		sleep 5; \
	done


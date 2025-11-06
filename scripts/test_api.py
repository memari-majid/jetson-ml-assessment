#!/usr/bin/env python3
"""
Test script for Jetson ML Inference API
"""

import requests
import json
import time
from pathlib import Path

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """Test health endpoint"""
        print("\n" + "="*60)
        print("Testing Health Endpoint")
        print("="*60)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_models_list(self):
        """Test models list endpoint"""
        print("\n" + "="*60)
        print("Testing Models List")
        print("="*60)
        
        try:
            response = requests.get(f"{self.base_url}/models")
            print(f"Status: {response.status_code}")
            print(f"Available models: {response.json()['models']}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_benchmark(self, model_name="mobilenet_v2", iterations=50):
        """Test benchmark endpoint"""
        print("\n" + "="*60)
        print(f"Benchmarking {model_name}")
        print("="*60)
        
        try:
            response = requests.get(
                f"{self.base_url}/benchmark/{model_name}",
                params={"iterations": iterations}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Benchmark completed")
                print(f"   Average time: {data['avg_time_ms']:.2f} ms")
                print(f"   Throughput: {data['throughput_fps']:.2f} FPS")
                print(f"   Device: {data['device']}")
                return True
            else:
                print(f"❌ Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_predict_dummy(self, model_name="mobilenet_v2"):
        """Test prediction with dummy image"""
        print("\n" + "="*60)
        print(f"Testing Prediction with {model_name}")
        print("="*60)
        
        try:
            # Create a dummy image
            from PIL import Image
            import io
            
            # Create random RGB image
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Send request
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/predict/{model_name}",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prediction successful")
                print(f"   Inference time: {data['inference_time_ms']:.2f} ms")
                print(f"   Top prediction: Class {data['predictions'][0]['class_id']}")
                print(f"   Confidence: {data['predictions'][0]['confidence']:.4f}")
                return True
            else:
                print(f"❌ Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  Jetson ML Inference API - Test Suite                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        results = {
            'health': self.test_health(),
            'models': self.test_models_list(),
            'benchmark': self.test_benchmark(),
            'predict': self.test_predict_dummy()
        }
        
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60)
        
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test:<20} {status}")
        
        total = len(results)
        passed = sum(results.values())
        print(f"\nTotal: {passed}/{total} tests passed")
        
        return all(results.values())

def main():
    import sys
    
    # Check if server is running
    print("Checking if API server is running...")
    try:
        response = requests.get("http://localhost:8000/")
        print("✅ Server is running")
    except:
        print("❌ Server not running")
        print("\nStart the server first:")
        print("  python3 inference_api.py")
        print("\nOr run in background:")
        print("  python3 inference_api.py &")
        sys.exit(1)
    
    # Run tests
    tester = APITester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Simple Inference API Server for Jetson
FastAPI-based REST API for ML model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import uvicorn
import time
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Jetson ML Inference API",
    description="High-performance ML inference API for NVIDIA Jetson",
    version="1.0.0"
)

# Global model storage
models_cache = {}
device = None

def initialize_models():
    """Initialize ML models on startup"""
    global device, models_cache
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("Loading models...")
    
    # MobileNet-v2 (lightweight, fast)
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.to(device)
    mobilenet.eval()
    models_cache['mobilenet_v2'] = mobilenet
    
    # ResNet-18 (balanced)
    resnet18 = models.resnet18(pretrained=True)
    resnet18.to(device)
    resnet18.eval()
    models_cache['resnet18'] = resnet18
    
    logger.info(f"Loaded {len(models_cache)} models")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    initialize_models()
    logger.info("Server ready for inference!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "device": str(device),
        "models_loaded": list(models_cache.keys()),
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models": {}
    }
    
    if torch.cuda.is_available():
        health_status["gpu_name"] = torch.cuda.get_device_name(0)
        health_status["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    for model_name in models_cache.keys():
        health_status["models"][model_name] = "loaded"
    
    return health_status

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(models_cache.keys()),
        "count": len(models_cache)
    }

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    """
    Run inference on uploaded image
    
    Args:
        model_name: Name of model to use (mobilenet_v2, resnet18)
        file: Image file (JPG, PNG)
    
    Returns:
        JSON with predictions
    """
    # Validate model
    if model_name not in models_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models_cache.keys())}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            output = models_cache[model_name](input_tensor)
        inference_time = time.time() - start_time
        
        # Get top predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(5):
            predictions.append({
                "class_id": int(top5_idx[i]),
                "confidence": float(top5_prob[i])
            })
        
        return {
            "model": model_name,
            "predictions": predictions,
            "inference_time_ms": inference_time * 1000,
            "device": str(device)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict/{model_name}")
async def batch_predict(model_name: str, files: List[UploadFile] = File(...)):
    """
    Run batch inference on multiple images
    
    Args:
        model_name: Name of model to use
        files: List of image files
    
    Returns:
        JSON with batch predictions
    """
    if model_name not in models_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        # Process all images
        batch_tensors = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            tensor = transform(image)
            batch_tensors.append(tensor)
        
        # Create batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Run batch inference
        start_time = time.time()
        with torch.no_grad():
            outputs = models_cache[model_name](batch)
        inference_time = time.time() - start_time
        
        # Process results
        results = []
        for i, output in enumerate(outputs):
            probabilities = torch.nn.functional.softmax(output, dim=0)
            top_prob, top_idx = torch.topk(probabilities, 3)
            
            results.append({
                "image_index": i,
                "top_class": int(top_idx[0]),
                "confidence": float(top_prob[0])
            })
        
        return {
            "model": model_name,
            "batch_size": len(files),
            "results": results,
            "total_inference_time_ms": inference_time * 1000,
            "avg_time_per_image_ms": (inference_time / len(files)) * 1000,
            "throughput_fps": len(files) / inference_time
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/{model_name}")
async def benchmark_model(model_name: str, iterations: int = 100):
    """
    Benchmark model performance
    
    Args:
        model_name: Name of model to benchmark
        iterations: Number of inference iterations
    
    Returns:
        Benchmark statistics
    """
    if model_name not in models_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = models_cache[model_name](dummy_input)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                _ = models_cache[model_name](dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        import numpy as np
        times = np.array(times)
        
        return {
            "model": model_name,
            "iterations": iterations,
            "avg_time_ms": float(np.mean(times) * 1000),
            "std_time_ms": float(np.std(times) * 1000),
            "min_time_ms": float(np.min(times) * 1000),
            "max_time_ms": float(np.max(times) * 1000),
            "throughput_fps": float(1.0 / np.mean(times)),
            "device": str(device)
        }
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the API server"""
    logger.info("Starting Jetson ML Inference API...")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()


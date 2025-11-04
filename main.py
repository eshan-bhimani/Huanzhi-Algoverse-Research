"""
Image Editing API using FastAPI and OpenCV
Supports image upload and URL-based processing with grayscale conversion
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
import cv2
import numpy as np
from typing import Optional
import requests
from io import BytesIO
import logging
from datetime import datetime
import json
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/requests_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Editing API",
    description="API for AI agent image editing tasks with CV2",
    version="0.1.0"
)


def log_request(prompt: str, image_source: str, operation: str, success: bool, error: str = None):
    """Log request details for later analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "image_source": image_source,
        "operation": operation,
        "success": success,
        "error": error
    }
    logger.info(json.dumps(log_entry))
    return log_entry


def load_image_from_url(url: str) -> np.ndarray:
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from URL")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file"""
    try:
        contents = file.file.read()
        image_array = np.asarray(bytearray(contents), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode uploaded image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load uploaded image: {str(e)}")


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale using CV2"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def encode_image(image: np.ndarray, format: str = '.png') -> bytes:
    """Encode image to bytes"""
    success, encoded_image = cv2.imencode(format, image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return encoded_image.tobytes()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "Image Editing API",
        "version": "0.1.0"
    }


@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Edit an image based on text prompt
    
    Args:
        prompt: Natural language description of editing task
        image_url: URL to image (optional)
        image_file: Uploaded image file (optional)
    
    Returns:
        Edited image as PNG
    """
    
    # Validate input
    if not image_url and not image_file:
        raise HTTPException(status_code=400, detail="Either image_url or image_file must be provided")
    
    if image_url and image_file:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_file, not both")
    
    # Load image
    try:
        if image_url:
            image = load_image_from_url(image_url)
            image_source = f"url:{image_url}"
        else:
            image = load_image_from_upload(image_file)
            image_source = f"upload:{image_file.filename}"
        
        # Simple prompt parsing for grayscale conversion
        # This is a minimal implementation - you'll expand this with more operations
        prompt_lower = prompt.lower()
        
        if "grayscale" in prompt_lower or "gray" in prompt_lower or "grey" in prompt_lower:
            edited_image = convert_to_grayscale(image)
            operation = "grayscale"
        else:
            # Default to grayscale for now
            edited_image = convert_to_grayscale(image)
            operation = "grayscale_default"
        
        # Log successful request
        log_request(prompt, image_source, operation, success=True)
        
        # Encode and return
        image_bytes = encode_image(edited_image)
        return Response(content=image_bytes, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        # Log failed request
        log_request(prompt, image_source if 'image_source' in locals() else "unknown", 
                   "error", success=False, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/logs")
async def get_logs(limit: int = 100):
    """
    Retrieve recent request logs
    
    Args:
        limit: Maximum number of logs to return
    """
    try:
        log_file = f'logs/requests_{datetime.now().strftime("%Y%m%d")}.log'
        if not os.path.exists(log_file):
            return {"logs": [], "message": "No logs found for today"}
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse JSON logs
        logs = []
        for line in lines[-limit:]:
            try:
                # Extract JSON from log line
                json_start = line.find('{')
                if json_start != -1:
                    log_data = json.loads(line[json_start:])
                    logs.append(log_data)
            except json.JSONDecodeError:
                continue
        
        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

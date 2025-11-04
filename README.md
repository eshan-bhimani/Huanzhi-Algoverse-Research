# Huanzhi-Algoverse-Research

# Image Editing API

A minimal FastAPI-based image editing pipeline using OpenCV (cv2) for benchmarking AI agents' tool-use capabilities in vision tasks.

## Overview

This API provides a REST interface for image editing operations with built-in evaluation metrics. Designed for research on multimodal AI agents that use tools for image manipulation tasks.

## Features

- 🖼️ **Dual Input Support**: Accept images via file upload or URL
- 🔧 **CV2 Integration**: OpenCV-based image processing
- 📊 **Built-in Evaluation**: MSE, PSNR, SSIM, and pixel accuracy metrics
- 📝 **Request Logging**: Automatic logging for analysis
- 🚀 **Easy Deployment**: Simple FastAPI setup

## Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd image-editing-api

# Install dependencies
pip install -r requirements.txt
```

### Running the Server
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Running the Demo
```bash
python demo.py
```

This will:
1. Create a test image
2. Send it to the API for grayscale conversion
3. Evaluate the results
4. Generate comparison images
5. Display logs

## API Endpoints

### `POST /edit`

Edit an image based on a text prompt.

**Parameters:**
- `prompt` (string, required): Natural language description of the editing task
- `image_url` (string, optional): URL to the image
- `image_file` (file, optional): Uploaded image file

**Note:** Provide either `image_url` OR `image_file`, not both.

**Example with cURL:**
```bash
# Upload file
curl -X POST "http://localhost:8000/edit" \
  -F "image_file=@test_image.jpg" \
  -F "prompt=Convert to grayscale" \
  --output result.png

# Use URL
curl -X POST "http://localhost:8000/edit" \
  -F "image_url=https://example.com/image.jpg" \
  -F "prompt=Convert to grayscale" \
  --output result.png
```

**Example with Python:**
```python
import requests

# Upload file
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/edit',
        files={'image_file': f},
        data={'prompt': 'Convert to grayscale'}
    )

# Save result
with open('result.png', 'wb') as f:
    f.write(response.content)
```

### `GET /logs`

Retrieve request logs for analysis.

**Parameters:**
- `limit` (int, optional): Maximum number of logs to return (default: 100)

**Example:**
```bash
curl "http://localhost:8000/logs?limit=10"
```

### `GET /`

Health check endpoint.

## Evaluation Module

The `evaluation.py` module provides comprehensive metrics for comparing edited images against ground truth:

### Metrics

- **MSE (Mean Squared Error)**: Pixel-level difference (lower is better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Quality measure in dB (higher is better)
- **SSIM (Structural Similarity Index)**: Perceptual similarity (0-1, higher is better)
- **Pixel Accuracy**: Percentage of exact pixel matches

### Usage Example
```python
from evaluation import evaluate_grayscale_conversion
import cv2

# Load images
original = cv2.imread('original.jpg')
edited = cv2.imread('edited.jpg', cv2.IMREAD_GRAYSCALE)

# Evaluate
result = evaluate_grayscale_conversion(original, edited)

print(f"MSE: {result.mse:.4f}")
print(f"PSNR: {result.psnr:.2f} dB")
print(f"SSIM: {result.ssim:.4f}")
print(f"Pixel Accuracy: {result.pixel_accuracy:.2f}%")
print(f"Passed: {result.passed}")
```

## Project Structure
```
image-editing-api/
├── main.py              # FastAPI application
├── evaluation.py        # Evaluation metrics
├── demo.py             # Demonstration script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── logs/              # Request logs (created automatically)
```

## Current Capabilities

### Supported Operations

- ✅ Grayscale conversion (deterministic, primary operation)
- 🚧 Additional operations to be added

### Planned Features

- [ ] Resize/crop operations
- [ ] Blur and filters
- [ ] Color adjustments
- [ ] Face detection
- [ ] Object segmentation
- [ ] Advanced prompt parsing
- [ ] Multi-step operation chains

## Development

### Adding New Operations

1. Add the operation function to `main.py`:
```python
def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

2. Update the prompt parsing in the `/edit` endpoint:
```python
if "blur" in prompt_lower:
    edited_image = apply_gaussian_blur(image)
    operation = "gaussian_blur"
```

3. Add evaluation function to `evaluation.py` if needed

### Testing
```python
# Run demo
python demo.py

# Check API documentation
# Navigate to http://localhost:8000/docs
```

## Logging

All requests are logged to `logs/requests_YYYYMMDD.log` with the following information:
- Timestamp
- Prompt
- Image source (URL or filename)
- Operation performed
- Success/failure status
- Error messages (if any)

## Performance Considerations

- Images are processed in memory
- No persistent storage of uploaded images
- Logs rotate daily
- Consider adding rate limiting for production use

## Contributing

This is a research project for benchmarking AI agents. Contributions welcome!

### Adding Operations

Focus on operations that:
1. Are deterministic (same input → same output)
2. Have clear success criteria
3. Are commonly requested in real-world scenarios
4. Can be objectively evaluated

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:
```
[Add citation once paper is published]
```

## Contact

bhimanieshan@gmail.com

## Acknowledgments

Part of research on vision-based tool use benchmarks for AI agents.

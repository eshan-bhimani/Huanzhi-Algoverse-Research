# GeoGuessr as a Vision Tool Use Benchmark

## Core Task
Agent receives a Street View image and must use visual analysis tools (not just guess) to systematically narrow down the location through observable features.

## Tool Use Aspect
Instead of end-to-end "look at image → output coordinates," the agent must:
- Use CV tools to detect/extract features (road signs, architecture, vegetation, license plates)
- Query knowledge about what these features mean geographically
- Iteratively refine hypothesis through systematic analysis
- Justify reasoning with visual evidence

## Environment Setup
- **Data Source**: Google Street View API or pre-collected dataset of Street View images with ground truth coordinates
- **Tool Library**: OpenCV for feature detection, OCR for text extraction, image classification models for architecture/vegetation/vehicle types
- **Output Format**: Predicted coordinates + reasoning trace showing which tools were used

## Evaluation Metrics
- **Distance Error**: Kilometers from ground truth (actual location/exact coordinates)
- **Score Bins**: Like actual GeoGuessr (5000 pts for <1km, scaling down)
- **Tool Use Effectiveness**: Did agent extract the "right" features?
- **Reasoning Quality**: Can it explain why it made its guess?

---

## Connecting Image Editing + GeoGuessr

### Shared Core: Vision-Based Tool Use for Analysis vs. Transformation

**Image Editing**: Given visual input + instruction → **transform** the image using algorithmic tools

**GeoGuessr**: Given visual input → **analyze** the image using algorithmic tools to extract actionable information

### Both Require:
1. **Visual perception**: Understanding what's in the image
2. **Tool selection**: Choosing appropriate CV operations
3. **Systematic execution**: Applying tools in logical sequence
4. **Verification**: Checking if the result meets the goal

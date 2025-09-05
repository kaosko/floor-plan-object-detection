# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Installation

**Choose one installation method:**

**CPU-only (servers, RunPod CPU):**
```bash
pip install -r requirements.txt
```

**GPU-accelerated (CUDA systems):**
```bash
pip install -r requirements-gpu.txt
```

**Run applications:**
- **Main object detection GUI**: `streamlit run app.py`
- **Dashed line detection GUI**: `python start_gui.py` (preferred) or `streamlit run dash_detection_gui.py`

### Development Commands
- **Start virtual environment** (if exists): `source venv/bin/activate`
- **Install new packages**: Add to `requirements.txt` and run `pip install -r requirements.txt`

## Architecture Overview

This is a dual-purpose floor plan analysis system with two main components:

### 1. Object Detection System (`app.py`)
- **Primary model**: YOLOv8 trained on architectural floor plan elements
- **Model file**: `best.pt` (52MB YOLOv8 model)
- **Key modules**:
  - `setting.py`: Streamlit UI configuration and user input handling
  - `helper.py`: Object counting and CSV export utilities
- **Features**: 
  - Tiled processing for large images (640x640 tiles)
  - Parallel processing support
  - Confidence threshold adjustment
  - Dynamic font sizing for annotations
  - CSV export of detection counts

### 2. Dashed Line Detection System
- **Main GUI**: `dash_detection_gui.py` with startup script `start_gui.py`
- **Core modules**:
  - `dash_line_detector.py`: Template-based dashed line detection
  - `integrated_dash_detector.py`: Complete detection pipeline
  - `robust_detectors.py`: Alternative detection methods
  - `scale_detector.py`: Scale and measurement detection
  - `legend_classifier.py`: Floor plan legend analysis
  - `output_exporter.py`: Multi-format result export
- **Features**:
  - Interactive ROI selection with drawable canvas
  - Template learning from user-drawn rectangles
  - Scale setting with line drawing
  - Multiple export formats (DXF, CSV, JSON)

### Key Technical Details

**Image Processing**:
- PIL image limit increased to 500M pixels for large floor plans
- PNG chunk limit set to 10MB for proper handling
- PyTorch safe-loading configured for YOLOv8 model compatibility

**Detection Pipeline**:
- Object detection uses 640x640 tiling with overlap handling
- Dashed line detection uses template matching and morphological operations
- Both systems support debug tile saving for troubleshooting

**UI Framework**: All interfaces built with Streamlit, dashed line GUI uses `streamlit-drawable-canvas` for interactive drawing

**Session State**: Both GUIs maintain state across reruns for uploaded images, detection results, and user configurations

## File Structure

- `app.py`: Main object detection Streamlit app
- `dash_detection_gui.py`: Dashed line detection interface
- `start_gui.py`: Startup script for dashed line GUI
- `best.pt`: Pre-trained YOLOv8 model for floor plan objects
- `requirements.txt`: Python dependencies including Streamlit, Ultralytics, OpenCV
- `*_detector.py`, `*_classifier.py`: Specialized detection modules
- Documentation: `README.md`, `README_DASH_DETECTION.md`, `GUI_FEATURES.md`, `MODEL_SELECTION_GUIDE.md`
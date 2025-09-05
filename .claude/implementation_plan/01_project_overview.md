# Analytical Symbol Detection System - Project Overview

## Purpose
Transform the existing analytical_symbol_detection.py script into a modular, class-based system with Streamlit UI for detecting symbols in PDF floor plans.

## Key Features
- PDF to PNG conversion with configurable zoom
- Interactive ROI selection for template learning
- Multi-scale and multi-angle template matching
- Coarse-to-fine detection strategy
- Export to YOLO format
- Symbol-specific detection classes
- Streamlit-based dashboard

## Architecture Principles
1. **Separation of Concerns**: Each class handles a specific responsibility
2. **Test-Driven Development**: Write tests before implementation
3. **Incremental Development**: Small, verifiable changes
4. **Configuration Management**: All parameters exposed via Streamlit UI
5. **Extensibility**: Easy to add new symbol types

## Technology Stack
- Python 3.8+
- Streamlit for UI
- OpenCV for image processing
- PyMuPDF for PDF handling
- NumPy for numerical operations
- pytest for testing

## Development Phases
1. Core infrastructure setup
2. Base classes implementation
3. Symbol detection classes
4. Streamlit UI components
5. Integration and testing
6. Performance optimization
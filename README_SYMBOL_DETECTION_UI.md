# Analytical Symbol Detection UI

This document describes the Phase 5 Streamlit User Interface for the Analytical Symbol Detection System.

## Overview

The Analytical Symbol Detection UI provides a modern, interactive web interface for floor plan symbol detection. It integrates with the existing `analytical_symbol_detection.py` backend to provide a user-friendly way to:

- Upload and process PDF floor plans
- Configure detection parameters interactively
- Select regions of interest (ROI) for template extraction
- Run symbol detection with real-time feedback
- Visualize and analyze detection results
- Export results in multiple formats

## Quick Start

### 1. Launch the UI

**Option 1: Using the startup script (recommended)**
```bash
python start_symbol_ui.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run symbol_detection_ui.py
```

The interface will open automatically in your web browser at `http://localhost:8501`.

### 2. Basic Workflow

1. **Upload PDF**: Use the sidebar to upload a floor plan PDF
2. **Configure Settings**: Adjust detection parameters as needed
3. **Select ROI**: Choose a region containing your target symbol
4. **Run Detection**: Click "Run Detection" to find symbols
5. **View Results**: Analyze results and export data

## Features

### Modern User Interface
- **Tabbed Interface**: Organized workflow with clear progression
- **Interactive Controls**: Sliders, dropdowns, and input fields
- **Real-time Feedback**: Progress indicators and status updates
- **Responsive Design**: Works on desktop and tablet devices

### Advanced Configuration
- **PDF Processing**: Page selection, zoom control
- **Detection Parameters**: Threshold, scales, angles, matching method
- **Performance Tuning**: Coarse search scale, top-K candidates
- **Symbol Classes**: HVAC, doors, windows, electrical, plumbing, custom

### Visualization & Analysis
- **Results Visualization**: Annotated images with bounding boxes
- **Detection Statistics**: Confidence distributions, area analysis
- **Template Preview**: Visual feedback on extracted templates
- **Processing Stages**: Step-by-step pipeline visualization

### Export Options
- **YOLO Format**: Standard object detection labels
- **CSV Export**: Detailed detection data
- **Annotated Images**: Visual results with overlays
- **Configuration Files**: Save/load detection settings

## Architecture

### Components

The Phase 5 UI consists of several modular components:

```
src/
├── ui/
│   ├── app.py              # Main Streamlit application
│   ├── components.py       # Reusable UI components
│   └── visualization.py    # Visualization utilities
├── core/
│   └── config_manager.py   # Configuration management
├── processors/
│   ├── pdf_processor.py    # PDF handling
│   ├── image_processor.py  # Image preprocessing
│   └── roi_selector.py     # ROI selection
└── detection/
    ├── data_classes.py     # Data structures
    ├── template_extractor.py # Template extraction
    └── detection_pipeline.py # Detection orchestration
```

### Integration

The UI integrates with the existing system through:

- **Backend Integration**: Uses `analytical_symbol_detection.py` as the detection engine
- **Configuration Bridge**: Maps UI settings to command-line arguments
- **Result Processing**: Loads and visualizes output from detection pipeline
- **File Management**: Handles temporary files and output directories

## User Guide

### PDF Processing Tab

**Upload PDF**
- Drag and drop or browse for PDF files
- Automatic file validation and preview
- Page count detection and selection

**Settings**
- **Page Number**: Select which page to analyze (0-based)
- **Zoom Factor**: Higher values = better quality, larger files
- **Preview**: Render PDF page for visual confirmation

### ROI Selection Tab

**Manual Entry**
- Enter X, Y coordinates for top-left corner
- Specify width and height in pixels
- Visual feedback on ROI location

**Interactive Selection** (planned)
- Click and drag to select regions
- Real-time preview of selected area
- Automatic validation and adjustment

**Template Analysis**
- Quality assessment of extracted template
- Feature analysis (edges, contours, contrast)
- Recommendations for improvement

### Detection Tab

**Configuration**
- **Threshold**: Sensitivity of detection (0.1-1.0)
- **Scales**: Size variations to search (e.g., 0.9,1.0,1.1)
- **Angles**: Rotation angles to test (degrees)
- **Method**: Template matching algorithm

**Advanced Settings**
- **Edge Detection**: Use edges vs. grayscale matching
- **Coarse Scale**: Downscale factor for initial search
- **Top-K**: Number of candidates to refine
- **Refinement Padding**: Search area expansion

**Execution**
- One-click detection with progress tracking
- Real-time status updates
- Error handling and recovery

### Results Tab

**Summary Statistics**
- Total detection count
- Average confidence scores
- Processing time metrics
- Quality assessments

**Visualization**
- Annotated images with bounding boxes
- Confidence heat maps
- Scale and angle distributions
- Detection density overlays

**Export Options**
- YOLO format labels for training
- CSV data for analysis
- High-resolution annotated images
- Processing logs and metadata

## Configuration

### Default Settings

The UI provides sensible defaults for common use cases:

**HVAC Symbols**
```python
threshold = 0.65
scales = [0.95, 1.0, 1.05]
angles = [0.0]
use_edges = True
method = "CCOEFF_NORMED"
```

**Doors/Windows**
```python
threshold = 0.70
scales = [0.9, 1.0, 1.1]
angles = [0.0, 90.0]
use_edges = True
```

**Electrical Components**
```python
threshold = 0.60
scales = [0.8, 0.9, 1.0, 1.1, 1.2]
angles = [0.0, 90.0, 180.0, 270.0]
use_edges = True
```

### Custom Configuration

Users can:
- Save configuration profiles for different symbol types
- Load previously saved settings
- Export configuration as JSON
- Share settings between team members

## Performance Optimization

### Speed vs. Accuracy Trade-offs

**Fast Detection (lower accuracy)**
- Coarse scale: 0.3-0.4
- Limited angles: [0.0]
- Higher threshold: 0.7+
- Fewer scales: [1.0]

**Thorough Detection (higher accuracy)**
- Coarse scale: 0.6-0.8
- Multiple angles: [0.0, 90.0, 180.0, 270.0]
- Lower threshold: 0.5-0.6
- Multiple scales: [0.8, 0.9, 1.0, 1.1, 1.2]

### Memory Management

- Automatic image resize for display
- Lazy loading of large images
- Temporary file cleanup
- Progress tracking for long operations

## Troubleshooting

### Common Issues

**No Detections Found**
- Lower the detection threshold
- Check template quality
- Try different scales or angles
- Verify ROI selection contains clear symbol

**Too Many False Positives**
- Increase detection threshold
- Improve template selection
- Use edge detection for line drawings
- Adjust NMS (non-maximum suppression) settings

**Slow Performance**
- Reduce PDF zoom factor
- Use smaller coarse search scale
- Limit number of search angles
- Select smaller ROI region

**UI Not Loading**
- Check Python version (3.8+ required)
- Verify all dependencies installed
- Check firewall/antivirus settings
- Try different web browser

### Error Messages

The UI provides helpful error messages and suggestions:

- **PDF Loading Errors**: File format validation, corruption detection
- **ROI Selection Errors**: Boundary checking, size validation
- **Detection Errors**: Parameter validation, memory issues
- **Export Errors**: File permissions, disk space

## Development

### Extending the UI

The modular architecture makes it easy to add new features:

**Adding New Visualization**
```python
# In src/ui/visualization.py
def create_custom_visualization(detections, image):
    # Custom visualization logic
    return annotated_image
```

**Adding New Export Format**
```python
# In src/ui/components.py
def export_custom_format(results):
    # Custom export logic
    return success_status
```

**Adding New Detection Method**
```python
# In src/detection/detection_pipeline.py
class CustomDetector(BaseDetector):
    def detect(self, image, template):
        # Custom detection logic
        return detections
```

### Testing

The UI includes comprehensive error handling and validation:

- Input parameter validation
- File format checking
- Memory usage monitoring
- Graceful error recovery

## API Reference

### Key Classes

**ConfigurationManager**
- Manages all detection parameters
- Validates configuration values
- Handles save/load operations

**DetectionPipeline** 
- Orchestrates detection workflow
- Manages processing stages
- Provides progress feedback

**VisualizationHelper**
- Creates annotated images
- Generates statistical plots
- Handles export operations

### Data Structures

**Detection**
```python
@dataclass
class Detection:
    x1: int           # Bounding box coordinates
    y1: int
    x2: int
    y2: int
    confidence: float # Detection confidence (0-1)
    class_name: str   # Symbol class
    scale: float      # Detection scale
    angle: float      # Detection angle
```

**DetectionResults**
```python
@dataclass 
class DetectionResults:
    detections: List[Detection]
    processing_time: float
    metadata: Dict[str, Any]
```

## Future Enhancements

### Planned Features
- **Interactive ROI Selection**: Canvas-based region drawing
- **Batch Processing**: Multiple PDFs at once  
- **Cloud Integration**: Remote processing capabilities
- **Advanced Analytics**: Statistical analysis tools
- **Custom Symbol Training**: User-defined symbol classes
- **Collaboration Features**: Shared workspaces and results

### Performance Improvements
- **GPU Acceleration**: CUDA-based template matching
- **Parallel Processing**: Multi-threaded detection
- **Caching System**: Intelligent result caching
- **Progressive Loading**: Stream processing for large files

## Support

For issues, questions, or feature requests:

1. Check the troubleshooting section above
2. Review the existing issues in the repository
3. Create a new issue with detailed description
4. Include configuration, error messages, and sample files

## License

This Phase 5 UI component follows the same license as the main project.
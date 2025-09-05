# Phase 6: Integration and Pipeline

## Objective
Integrate all components into a cohesive detection pipeline and ensure smooth data flow.

## Tasks

### Task 6.1: Detection Pipeline
**File**: `src/detection/detection_pipeline.py`

**Implementation**:
```python
import time
import os
from typing import Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path

from src.models.data_models import DetectionConfig, DetectionResults, Detection
from src.processors.pdf_processor import PDFProcessor
from src.processors.image_processor import ImagePreprocessor
from src.processors.roi_selector import ROISelector
from src.detection.template_extractor import TemplateExtractor
from src.detection.matching_engine import TemplateMatchingEngine
from src.detection.base_detector import BaseSymbolDetector
from src.detection.hvac_detector import HVACSymbolDetector
from src.detection.door_detector import DoorSymbolDetector
from src.detection.window_detector import WindowSymbolDetector

class DetectionPipeline:
    """Orchestrates the complete detection pipeline"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.pdf_processor = None
        self.image_processor = ImagePreprocessor()
        self.roi_selector = ROISelector()
        self.template_extractor = TemplateExtractor()
        self.detector = None
        
        # Initialize appropriate detector based on class name
        self._initialize_detector()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_detector(self):
        """Initialize the appropriate symbol detector"""
        detector_map = {
            'hvac': HVACSymbolDetector,
            'door': DoorSymbolDetector,
            'window': WindowSymbolDetector,
            'hvac_symbol': HVACSymbolDetector,
        }
        
        detector_class = detector_map.get(
            self.config.class_name.lower(),
            BaseSymbolDetector
        )
        
        if detector_class == BaseSymbolDetector:
            # For base class, need to provide concrete implementation
            class GenericDetector(BaseSymbolDetector):
                def detect(self, image, template):
                    matches = self.matching_engine.coarse_to_fine_match(
                        image, template,
                        self.config.coarse_scale,
                        self.config.scales,
                        self.config.angles,
                        self.threshold,
                        self.config.topk,
                        self.config.refine_pad
                    )
                    detections = self.matches_to_detections(matches)
                    return self.apply_nms(detections)
                
                def get_detection_params(self):
                    return {'generic': True}
            
            self.detector = GenericDetector(
                self.config.class_name,
                self.config.threshold,
                self.config
            )
        else:
            self.detector = detector_class(self.config.threshold)
            self.detector.config = self.config
    
    def run(self, image: Optional[np.ndarray] = None, 
           template: Optional[np.ndarray] = None) -> DetectionResults:
        """Run the complete detection pipeline"""
        start_time = time.time()
        
        # Step 1: Load and render PDF if needed
        if image is None:
            image = self._load_image()
        
        # Step 2: Get or select ROI
        if template is None:
            template = self._get_template(image)
        
        # Step 3: Preprocess image if needed
        if self.config.use_edges:
            search_image = self.image_processor.extract_edges(image)
            template_processed = self.image_processor.extract_edges(template)
        else:
            search_image = self.image_processor.convert_to_grayscale(image)
            template_processed = self.image_processor.convert_to_grayscale(template)
        
        # Step 4: Run detection
        detections = self.detector.detect(search_image, template_processed)
        
        # Step 5: Create results
        processing_time = time.time() - start_time
        
        metadata = {
            'config': self.config.to_dict(),
            'image_shape': image.shape,
            'template_shape': template.shape,
            'detector_type': self.detector.__class__.__name__,
            'detector_params': self.detector.get_detection_params()
        }
        
        results = DetectionResults(
            detections=detections,
            processing_time=processing_time,
            metadata=metadata
        )
        
        return results
    
    def _load_image(self) -> np.ndarray:
        """Load image from PDF"""
        # Check if cached image exists
        cache_path = self.output_dir / f"page{self.config.page}.png"
        
        if cache_path.exists():
            print(f"Loading cached image: {cache_path}")
            image = cv2.imread(str(cache_path))
        else:
            print(f"Rendering PDF page {self.config.page}...")
            self.pdf_processor = PDFProcessor(
                self.config.pdf_path,
                self.config.page,
                self.config.zoom
            )
            self.pdf_processor.render_to_png(str(cache_path))
            image = cv2.imread(str(cache_path))
        
        if image is None:
            raise RuntimeError(f"Failed to load image from {cache_path}")
        
        return image
    
    def _get_template(self, image: np.ndarray) -> np.ndarray:
        """Get template from ROI"""
        # Check if ROI is provided
        if self.config.roi:
            roi = self.config.roi
        else:
            # Check for saved ROI
            roi_path = self.output_dir / "roi.json"
            if roi_path.exists():
                roi = self.roi_selector.load_roi(str(roi_path))
            else:
                # Interactive selection
                roi = self.roi_selector.select_interactive(image)
                self.roi_selector.save_roi(roi, str(roi_path))
        
        # Extract template
        template = self.template_extractor.extract_template(image, roi)
        
        # Optional: Apply interior mask and tighten ROI
        mask = self.template_extractor.create_interior_mask(template)
        if np.any(mask > 0):
            roi = self.template_extractor.tighten_roi(
                roi, mask, image.shape[:2]
            )
            template = self.template_extractor.extract_template(image, roi)
        
        # Save template for reference
        template_path = self.output_dir / "template.png"
        cv2.imwrite(str(template_path), template)
        
        return template
    
    def save_results(self, results: DetectionResults, 
                    image: Optional[np.ndarray] = None) -> Dict[str, str]:
        """Save detection results in various formats"""
        output_files = {}
        
        # Save YOLO labels
        labels_path = self.output_dir / f"page{self.config.page}.txt"
        if image is not None:
            self.detector.export_yolo_format(
                results.detections,
                image.shape[:2],
                str(labels_path)
            )
            output_files['yolo_labels'] = str(labels_path)
        
        # Save class names
        classes_path = self.output_dir / "classes.txt"
        with open(classes_path, 'w') as f:
            f.write(self.config.class_name + "\n")
        output_files['classes'] = str(classes_path)
        
        # Save detections as JSON
        import json
        json_path = self.output_dir / "detections.json"
        detections_dict = [
            {
                'x1': d.x1, 'y1': d.y1,
                'x2': d.x2, 'y2': d.y2,
                'confidence': d.confidence,
                'class': d.class_name,
                'scale': d.scale,
                'angle': d.angle
            }
            for d in results.detections
        ]
        
        with open(json_path, 'w') as f:
            json.dump({
                'detections': detections_dict,
                'metadata': results.metadata,
                'processing_time': results.processing_time
            }, f, indent=2)
        output_files['json'] = str(json_path)
        
        # Save annotated image if provided
        if image is not None:
            annotated = self._draw_annotations(image, results.detections)
            annotated_path = self.output_dir / "annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            output_files['annotated_image'] = str(annotated_path)
        
        # Save summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Detection Summary\n")
            f.write(f"================\n\n")
            f.write(f"Total detections: {len(results.detections)}\n")
            f.write(f"Processing time: {results.processing_time:.2f}s\n")
            f.write(f"Class: {self.config.class_name}\n")
            f.write(f"Threshold: {self.config.threshold}\n")
            f.write(f"Scales: {self.config.scales}\n")
            f.write(f"Angles: {self.config.angles}\n")
            
            if results.detections:
                confidences = [d.confidence for d in results.detections]
                f.write(f"\nConfidence stats:\n")
                f.write(f"  Min: {min(confidences):.3f}\n")
                f.write(f"  Max: {max(confidences):.3f}\n")
                f.write(f"  Mean: {np.mean(confidences):.3f}\n")
                f.write(f"  Std: {np.std(confidences):.3f}\n")
        
        output_files['summary'] = str(summary_path)
        
        return output_files
    
    def _draw_annotations(self, image: np.ndarray, 
                         detections: list[Detection]) -> np.ndarray:
        """Draw detection boxes on image"""
        annotated = image.copy()
        
        for det in detections:
            # Draw rectangle
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                (0, 255, 0),
                2
            )
            
            # Draw label
            label = f"{det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Background for label
            cv2.rectangle(
                annotated,
                (det.x1, max(0, det.y1 - label_size[1] - 4)),
                (det.x1 + label_size[0], det.y1),
                (0, 255, 0),
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label,
                (det.x1, max(label_size[1], det.y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated
```

**Tests**:
```python
# tests/test_detection_pipeline.py
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from src.detection.detection_pipeline import DetectionPipeline
from src.models.data_models import DetectionConfig

@pytest.fixture
def test_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DetectionConfig(
            pdf_path='test.pdf',  # Would need actual test PDF
            output_dir=tmpdir,
            class_name='test_symbol'
        )
        yield config

@pytest.fixture
def test_image():
    # Create test image
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (150, 150), (0, 0, 0), -1)
    cv2.rectangle(img, (300, 300), (350, 350), (0, 0, 0), -1)
    return img

@pytest.fixture
def test_template():
    # Create test template
    template = np.ones((50, 50, 3), dtype=np.uint8) * 255
    cv2.rectangle(template, (10, 10), (40, 40), (0, 0, 0), -1)
    return template

def test_pipeline_initialization(test_config):
    pipeline = DetectionPipeline(test_config)
    
    assert pipeline.config == test_config
    assert pipeline.detector is not None
    assert Path(pipeline.output_dir).exists()

def test_pipeline_run_with_inputs(test_config, test_image, test_template):
    pipeline = DetectionPipeline(test_config)
    
    results = pipeline.run(image=test_image, template=test_template)
    
    assert results is not None
    assert hasattr(results, 'detections')
    assert hasattr(results, 'processing_time')
    assert results.processing_time > 0

def test_save_results(test_config, test_image, test_template):
    pipeline = DetectionPipeline(test_config)
    results = pipeline.run(image=test_image, template=test_template)
    
    output_files = pipeline.save_results(results, test_image)
    
    assert 'yolo_labels' in output_files
    assert 'json' in output_files
    assert 'summary' in output_files
    
    # Check files exist
    for filepath in output_files.values():
        assert Path(filepath).exists()

def test_detector_selection(test_config):
    # Test HVAC detector
    test_config.class_name = 'hvac'
    pipeline = DetectionPipeline(test_config)
    assert pipeline.detector.__class__.__name__ == 'HVACSymbolDetector'
    
    # Test door detector
    test_config.class_name = 'door'
    pipeline = DetectionPipeline(test_config)
    assert pipeline.detector.__class__.__name__ == 'DoorSymbolDetector'
    
    # Test generic detector
    test_config.class_name = 'custom'
    pipeline = DetectionPipeline(test_config)
    assert 'GenericDetector' in pipeline.detector.__class__.__name__
```

### Task 6.2: Command-Line Interface
**File**: `src/cli.py`

**Implementation**:
```python
#!/usr/bin/env python3
"""
Command-line interface for analytical symbol detection.
Provides argparse compatibility for the Streamlit application.
"""

import argparse
import sys
import json
from pathlib import Path

from src.models.data_models import DetectionConfig
from src.detection.detection_pipeline import DetectionPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analytical Symbol Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--pdf", 
        required=True,
        help="Path to input PDF file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--page", 
        type=int, 
        default=0,
        help="Zero-based page index"
    )
    
    parser.add_argument(
        "--zoom", 
        type=float, 
        default=6.0,
        help="PDF rasterization zoom factor"
    )
    
    parser.add_argument(
        "--outdir", 
        default="detection_output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.65,
        help="Detection threshold (0-1)"
    )
    
    parser.add_argument(
        "--scales", 
        default="0.95,1.0,1.05",
        help="Comma-separated scale factors"
    )
    
    parser.add_argument(
        "--angles", 
        default="0",
        help="Comma-separated angles in degrees"
    )
    
    parser.add_argument(
        "--method",
        choices=['CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'],
        default="CCOEFF_NORMED",
        help="Template matching method"
    )
    
    parser.add_argument(
        "--class-name", 
        default="object",
        help="Symbol class name"
    )
    
    parser.add_argument(
        "--no-edges",
        dest="use_edges",
        action="store_false",
        help="Disable edge detection"
    )
    
    parser.add_argument(
        "--roi",
        type=str,
        help="ROI as x,y,w,h (skip interactive selection)"
    )
    
    parser.add_argument(
        "--reuse-roi",
        action="store_true",
        help="Reuse saved ROI if available"
    )
    
    parser.add_argument(
        "--coarse",
        type=float,
        default=0.5,
        help="Coarse search scale (0-1)"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Top-K coarse candidates"
    )
    
    parser.add_argument(
        "--refine-pad",
        type=float,
        default=0.5,
        help="Refinement padding ratio"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Streamlit GUI instead of CLI"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from JSON file"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to JSON file"
    )
    
    return parser.parse_args()

def args_to_config(args) -> DetectionConfig:
    """Convert argparse namespace to DetectionConfig"""
    # Parse scales and angles
    scales = [float(s.strip()) for s in args.scales.split(',') if s.strip()]
    angles = [float(a.strip()) for a in args.angles.split(',') if a.strip()]
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        parts = [int(v) for v in args.roi.split(',')]
        if len(parts) == 4:
            roi = tuple(parts)
    
    config = DetectionConfig(
        pdf_path=args.pdf,
        page=args.page,
        zoom=args.zoom,
        threshold=args.threshold,
        scales=scales,
        angles=angles,
        method=args.method,
        class_name=args.class_name,
        use_edges=args.use_edges,
        roi=roi,
        coarse_scale=args.coarse,
        topk=args.topk,
        refine_pad=args.refine_pad,
        output_dir=args.outdir
    )
    
    return config

def main():
    args = parse_args()
    
    # Launch GUI if requested
    if args.gui:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "src/ui/app.py"]
        sys.exit(stcli.main())
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DetectionConfig(**config_dict)
    else:
        config = args_to_config(args)
    
    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Save config if requested
    if args.save_config:
        from src.core.config_manager import ConfigurationManager
        manager = ConfigurationManager()
        manager.config = config
        manager.save_to_file(args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Run detection pipeline
    print("Starting detection pipeline...")
    print(f"PDF: {config.pdf_path}")
    print(f"Page: {config.page}")
    print(f"Class: {config.class_name}")
    print(f"Output: {config.output_dir}")
    print()
    
    pipeline = DetectionPipeline(config)
    
    try:
        results = pipeline.run()
        
        print(f"\nDetection complete!")
        print(f"Found {len(results.detections)} instances")
        print(f"Processing time: {results.processing_time:.2f}s")
        
        # Save results
        output_files = pipeline.save_results(results)
        
        print("\nOutput files:")
        for file_type, filepath in output_files.items():
            print(f"  {file_type}: {filepath}")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Deliverables
1. Complete detection pipeline orchestrating all components
2. CLI with full argparse compatibility
3. Seamless integration between components
4. Result saving in multiple formats
5. Error handling and validation

## Success Criteria
- Pipeline runs end-to-end successfully
- CLI arguments match original script
- Can launch GUI from CLI
- Results saved correctly in all formats
- Proper error messages and handling
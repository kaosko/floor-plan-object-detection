# Incremental Implementation Milestones

## Overview
Small, verifiable milestones for incremental development and testing.

## Milestone 1: Project Foundation (Day 1-2)
**Goal**: Set up project structure and core data models

### Tasks:
1. Create project directory structure
2. Set up virtual environment and dependencies
3. Implement data models (Detection, DetectionConfig, DetectionResults)
4. Write and pass all data model tests
5. Set up pytest configuration and coverage reporting

### Verification:
- [ ] All imports work correctly
- [ ] Data model tests pass (100% coverage)
- [ ] Project structure matches specification
- [ ] CI/CD pipeline configured

### Deliverable:
```bash
pytest tests/unit/test_data_models.py -v
# All tests should pass
```

---

## Milestone 2: Configuration Management (Day 3)
**Goal**: Implement configuration loading/saving

### Tasks:
1. Implement ConfigurationManager class
2. Add JSON serialization/deserialization
3. Add argparse compatibility methods
4. Write configuration tests
5. Create sample configuration files

### Verification:
- [ ] Can save/load configurations
- [ ] Argparse conversion works
- [ ] Configuration validation works
- [ ] Tests pass with >90% coverage

### Deliverable:
```python
# Demo script
from src.core.config_manager import ConfigurationManager

manager = ConfigurationManager()
manager.config.pdf_path = "test.pdf"
manager.save_to_file("config.json")
manager.load_from_file("config.json")
print("Config loaded:", manager.config.pdf_path)
```

---

## Milestone 3: PDF Processing (Day 4-5)
**Goal**: PDF to image conversion working

### Tasks:
1. Implement PDFProcessor class
2. Add page rendering methods
3. Add page information methods
4. Write PDF processor tests
5. Test with sample PDFs

### Verification:
- [ ] Can render PDF pages to PNG
- [ ] Can get page count and dimensions
- [ ] Error handling for invalid PDFs
- [ ] Tests pass with fixtures

### Deliverable:
```python
# Demo script
from src.processors.pdf_processor import PDFProcessor

processor = PDFProcessor("sample.pdf", page=0, zoom=4.0)
processor.render_to_png("output.png")
print(f"Rendered page to output.png")
print(f"Total pages: {processor.get_page_count()}")
```

---

## Milestone 4: Image Processing (Day 6-7)
**Goal**: Image preprocessing capabilities

### Tasks:
1. Implement ImagePreprocessor class
2. Add grayscale, edge, binary methods
3. Add morphological operations
4. Write image processing tests
5. Create visual test outputs

### Verification:
- [ ] All preprocessing methods work
- [ ] Can handle color and grayscale images
- [ ] Edge detection produces valid results
- [ ] Tests pass with sample images

### Deliverable:
```python
# Demo script showing preprocessing pipeline
from src.processors.image_processor import ImagePreprocessor
import cv2

processor = ImagePreprocessor()
img = cv2.imread("sample.png")
gray = processor.convert_to_grayscale(img)
edges = processor.extract_edges(img)
binary = processor.binarize(img)

cv2.imwrite("gray.png", gray)
cv2.imwrite("edges.png", edges)
cv2.imwrite("binary.png", binary)
print("Preprocessing complete")
```

---

## Milestone 5: ROI Selection (Day 8-9)
**Goal**: ROI selection and management

### Tasks:
1. Implement ROISelector class
2. Add interactive selection (mock for testing)
3. Add ROI save/load functionality
4. Add ROI validation and expansion
5. Write ROI selector tests

### Verification:
- [ ] Can save/load ROI coordinates
- [ ] ROI validation works correctly
- [ ] Can extract ROI from image
- [ ] Tests pass with mock interactions

### Deliverable:
```python
# Demo script
from src.processors.roi_selector import ROISelector
import numpy as np

selector = ROISelector()
test_image = np.zeros((500, 500, 3), dtype=np.uint8)

# Manual ROI for testing
roi = (100, 100, 200, 200)
selector.save_roi(roi, "roi.json")
loaded_roi = selector.load_roi("roi.json")

roi_image = selector.extract_roi(test_image, loaded_roi)
print(f"ROI extracted: {roi_image.shape}")
```

---

## Milestone 6: Template Extraction (Day 10-11)
**Goal**: Template extraction and analysis

### Tasks:
1. Implement TemplateExtractor class
2. Add interior mask generation
3. Add ROI tightening functionality
4. Add template feature analysis
5. Write template extractor tests

### Verification:
- [ ] Can extract templates from ROI
- [ ] Interior mask generation works
- [ ] Feature analysis provides metrics
- [ ] Tests pass with sample templates

### Deliverable:
```python
# Demo script
from src.detection.template_extractor import TemplateExtractor
import cv2

extractor = TemplateExtractor()
img = cv2.imread("sample.png")
roi = (100, 100, 50, 50)

template = extractor.extract_template(img, roi)
mask = extractor.create_interior_mask(template)
features = extractor.analyze_template_features(template)

print(f"Template extracted: {template.shape}")
print(f"Features: {features}")
cv2.imwrite("template.png", template)
cv2.imwrite("mask.png", mask)
```

---

## Milestone 7: Template Matching Engine (Day 12-14)
**Goal**: Core matching algorithms working

### Tasks:
1. Implement TemplateMatchingEngine class
2. Add single-scale matching
3. Add multi-scale matching
4. Add multi-angle matching
5. Add coarse-to-fine matching
6. Write matching engine tests

### Verification:
- [ ] Single-scale matching finds patterns
- [ ] Multi-scale handles size variations
- [ ] Multi-angle handles rotations
- [ ] Coarse-to-fine improves performance
- [ ] All tests pass

### Deliverable:
```python
# Demo script
from src.detection.matching_engine import TemplateMatchingEngine
import cv2

engine = TemplateMatchingEngine('CCOEFF_NORMED')
scene = cv2.imread("scene.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

# Test different matching modes
matches_single = engine.match_single(scene, template, threshold=0.7)
matches_multi = engine.match_multi_scale(
    scene, template, [0.9, 1.0, 1.1], threshold=0.7
)

print(f"Single-scale: {len(matches_single)} matches")
print(f"Multi-scale: {len(matches_multi)} matches")
```

---

## Milestone 8: Base Detector (Day 15-16)
**Goal**: Base detector class with NMS

### Tasks:
1. Implement BaseSymbolDetector abstract class
2. Add NMS implementation
3. Add YOLO export functionality
4. Add filtering methods
5. Write base detector tests

### Verification:
- [ ] NMS removes duplicates correctly
- [ ] YOLO export format is valid
- [ ] Filtering methods work
- [ ] Abstract methods enforced
- [ ] Tests pass

### Deliverable:
```python
# Demo script with concrete implementation
from src.detection.base_detector import BaseSymbolDetector

class TestDetector(BaseSymbolDetector):
    def detect(self, image, template):
        # Simple implementation for testing
        matches = self.matching_engine.match_single(
            image, template, self.threshold
        )
        return self.matches_to_detections(matches)
    
    def get_detection_params(self):
        return {'test': True}

detector = TestDetector('test_symbol', threshold=0.7)
print(f"Detector created: {detector.class_name}")
```

---

## Milestone 9: Symbol-Specific Detectors (Day 17-19)
**Goal**: Implement HVAC, Door, Window detectors

### Tasks:
1. Implement HVACSymbolDetector
2. Implement DoorSymbolDetector
3. Implement WindowSymbolDetector
4. Add symbol-specific logic
5. Write detector tests

### Verification:
- [ ] Each detector has unique behavior
- [ ] Symbol-specific parameters work
- [ ] All detectors inherit properly
- [ ] Tests pass for each detector

### Deliverable:
```python
# Demo script testing each detector
from src.detection.hvac_detector import HVACSymbolDetector
from src.detection.door_detector import DoorSymbolDetector
from src.detection.window_detector import WindowSymbolDetector

detectors = [
    HVACSymbolDetector(),
    DoorSymbolDetector(),
    WindowSymbolDetector()
]

for detector in detectors:
    params = detector.get_detection_params()
    print(f"{detector.class_name}: {params}")
```

---

## Milestone 10: Detection Pipeline (Day 20-21)
**Goal**: Complete pipeline integration

### Tasks:
1. Implement DetectionPipeline class
2. Integrate all components
3. Add result saving methods
4. Add visualization methods
5. Write pipeline tests

### Verification:
- [ ] Pipeline runs end-to-end
- [ ] All components integrate smoothly
- [ ] Results saved in multiple formats
- [ ] Visualization works correctly
- [ ] Integration tests pass

### Deliverable:
```python
# Demo script - complete detection
from src.detection.detection_pipeline import DetectionPipeline
from src.models.data_models import DetectionConfig

config = DetectionConfig(
    pdf_path="sample.pdf",
    page=0,
    zoom=4.0,
    threshold=0.7,
    class_name="door",
    roi=(100, 100, 50, 50)
)

pipeline = DetectionPipeline(config)
results = pipeline.run()

print(f"Found {len(results.detections)} detections")
print(f"Time: {results.processing_time:.2f}s")

output_files = pipeline.save_results(results)
print(f"Results saved: {output_files}")
```

---

## Milestone 11: CLI Implementation (Day 22)
**Goal**: Command-line interface with argparse

### Tasks:
1. Implement CLI with argparse
2. Add config file support
3. Add GUI launch option
4. Write CLI tests
5. Create usage documentation

### Verification:
- [ ] All argparse parameters work
- [ ] Config save/load works
- [ ] Can launch GUI
- [ ] Error handling works
- [ ] Help text is clear

### Deliverable:
```bash
# Test CLI
python src/cli.py --pdf sample.pdf --page 0 --threshold 0.7 --class-name door --roi 100,100,50,50 --outdir output/

# Should output:
# Detection complete!
# Found X instances
# Results saved to output/
```

---

## Milestone 12: Streamlit UI - Basic (Day 23-24)
**Goal**: Basic Streamlit interface

### Tasks:
1. Implement main app structure
2. Add PDF upload functionality
3. Add configuration sidebar
4. Add image display
5. Test UI components

### Verification:
- [ ] Can upload PDF
- [ ] Configuration controls work
- [ ] Image displays correctly
- [ ] Layout is responsive
- [ ] No Streamlit errors

### Deliverable:
```bash
# Launch Streamlit app
streamlit run src/ui/app.py

# Should show:
# - File upload widget
# - Configuration sidebar
# - Image display area
```

---

## Milestone 13: Streamlit UI - ROI Selection (Day 25)
**Goal**: ROI selection in Streamlit

### Tasks:
1. Add ROI selection tab
2. Implement manual ROI input
3. Add ROI save/load
4. Add template display
5. Test ROI functionality

### Verification:
- [ ] Manual ROI input works
- [ ] ROI save/load works
- [ ] Template displays correctly
- [ ] Template features shown
- [ ] State management works

### Deliverable:
```python
# In Streamlit UI:
# 1. Upload PDF
# 2. Go to ROI Selection tab
# 3. Enter ROI manually
# 4. See extracted template
# 5. See template features
```

---

## Milestone 14: Streamlit UI - Detection (Day 26)
**Goal**: Detection functionality in UI

### Tasks:
1. Add detection tab
2. Add run detection button
3. Add progress indicators
4. Add result preview
5. Test detection flow

### Verification:
- [ ] Detection runs from UI
- [ ] Progress shown correctly
- [ ] Results display properly
- [ ] Annotations visible
- [ ] No UI freezing

### Deliverable:
```python
# In Streamlit UI:
# 1. Complete ROI selection
# 2. Go to Detection tab
# 3. Click Run Detection
# 4. See progress bar
# 5. See annotated results
```

---

## Milestone 15: Streamlit UI - Results (Day 27)
**Goal**: Results export and visualization

### Tasks:
1. Add results tab
2. Add statistics display
3. Add export buttons
4. Add results table
5. Test export functionality

### Verification:
- [ ] Statistics display correctly
- [ ] All export formats work
- [ ] Results table shows data
- [ ] Downloads work properly
- [ ] Visualization is clear

### Deliverable:
```python
# In Streamlit UI:
# 1. After detection
# 2. Go to Results tab
# 3. See statistics
# 4. Export YOLO labels
# 5. Export CSV
# 6. Save annotated image
```

---

## Milestone 16: End-to-End Testing (Day 28-29)
**Goal**: Complete system testing

### Tasks:
1. Write E2E test scenarios
2. Test CLI workflow
3. Test GUI workflow
4. Performance testing
5. Create test report

### Verification:
- [ ] CLI E2E tests pass
- [ ] GUI E2E tests pass
- [ ] Performance acceptable
- [ ] No memory leaks
- [ ] Coverage >85%

### Deliverable:
```bash
# Run full test suite
pytest tests/ --cov=src --cov-report=html

# Generate report
python -m pytest --html=report.html --self-contained-html
```

---

## Milestone 17: Documentation (Day 30)
**Goal**: Complete documentation

### Tasks:
1. Write README.md
2. Create API documentation
3. Write user guide
4. Add code comments
5. Create demo videos

### Verification:
- [ ] README is comprehensive
- [ ] API docs generated
- [ ] User guide complete
- [ ] Code well-commented
- [ ] Examples work

### Deliverable:
```bash
# Generate documentation
sphinx-build -b html docs/ docs/_build/

# Should create:
# - Full API reference
# - User guide
# - Installation instructions
# - Examples
```

---

## Final Verification Checklist

### Functionality
- [ ] PDF rendering works
- [ ] ROI selection works
- [ ] Template extraction works
- [ ] Detection works for all symbol types
- [ ] Results export works
- [ ] CLI fully functional
- [ ] GUI fully functional

### Quality
- [ ] All tests pass
- [ ] Code coverage >85%
- [ ] No linting errors
- [ ] Documentation complete
- [ ] Performance acceptable

### Compatibility
- [ ] Works with original argparse parameters
- [ ] YOLO format compatible
- [ ] Streamlit UI responsive
- [ ] Cross-platform compatible

## Success Metrics

1. **Functionality**: 100% of original features implemented
2. **Testing**: >85% code coverage
3. **Performance**: <5s for typical detection task
4. **Usability**: Both CLI and GUI functional
5. **Documentation**: Complete user and developer docs
6. **Maintainability**: Clean, modular architecture
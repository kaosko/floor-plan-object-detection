# Test-Driven Development Strategy

## Overview
Comprehensive testing strategy ensuring each component is thoroughly tested before integration.

## Testing Principles

1. **Write Tests First**: Define expected behavior before implementation
2. **Unit Tests**: Test individual methods and classes in isolation
3. **Integration Tests**: Test component interactions
4. **End-to-End Tests**: Test complete workflows
5. **Coverage Target**: Minimum 85% code coverage

## Test Structure

```
tests/
├── unit/
│   ├── test_data_models.py
│   ├── test_config_manager.py
│   ├── test_pdf_processor.py
│   ├── test_image_processor.py
│   ├── test_roi_selector.py
│   ├── test_template_extractor.py
│   ├── test_matching_engine.py
│   ├── test_base_detector.py
│   └── test_symbol_detectors.py
├── integration/
│   ├── test_detection_pipeline.py
│   ├── test_cli.py
│   └── test_streamlit_components.py
├── e2e/
│   ├── test_full_workflow.py
│   └── test_gui_workflow.py
├── fixtures/
│   ├── sample_pdfs/
│   ├── sample_images/
│   └── sample_templates/
└── conftest.py
```

## Testing Phases

### Phase 1: Core Infrastructure Tests
**Before implementing any core classes**

```python
# tests/unit/test_data_models.py
import pytest
from src.models.data_models import Detection, DetectionConfig, DetectionResults

class TestDetection:
    def test_creation(self):
        """Test Detection object creation"""
        det = Detection(10, 20, 100, 200, 0.95, 'door')
        assert det.x1 == 10
        assert det.confidence == 0.95
    
    def test_properties(self):
        """Test computed properties"""
        det = Detection(0, 0, 100, 50, 0.9, 'window')
        assert det.center == (50, 25)
        assert det.area == 5000
    
    def test_bbox(self):
        """Test bbox property"""
        det = Detection(10, 20, 30, 40, 0.8, 'hvac')
        assert det.bbox == (10, 20, 30, 40)

class TestDetectionConfig:
    def test_default_values(self):
        """Test default configuration values"""
        config = DetectionConfig(pdf_path='test.pdf')
        assert config.page == 0
        assert config.zoom == 6.0
        assert config.threshold == 0.65
    
    def test_validation_valid(self):
        """Test valid configuration"""
        config = DetectionConfig(pdf_path='test.pdf', zoom=5.0)
        assert config.validate() == True
    
    def test_validation_invalid_zoom(self):
        """Test invalid zoom value"""
        config = DetectionConfig(pdf_path='test.pdf', zoom=-1)
        with pytest.raises(ValueError, match="Zoom must be positive"):
            config.validate()
    
    def test_validation_invalid_threshold(self):
        """Test invalid threshold value"""
        config = DetectionConfig(pdf_path='test.pdf', threshold=1.5)
        with pytest.raises(ValueError, match="Threshold must be between"):
            config.validate()

class TestDetectionResults:
    def test_filter_by_confidence(self):
        """Test filtering by confidence threshold"""
        detections = [
            Detection(0, 0, 10, 10, 0.9, 'door'),
            Detection(0, 0, 10, 10, 0.5, 'window'),
            Detection(0, 0, 10, 10, 0.7, 'door')
        ]
        results = DetectionResults(detections, 1.0)
        filtered = results.filter_by_confidence(0.6)
        
        assert len(filtered.detections) == 2
        assert all(d.confidence >= 0.6 for d in filtered.detections)
    
    def test_filter_by_class(self):
        """Test filtering by class name"""
        detections = [
            Detection(0, 0, 10, 10, 0.9, 'door'),
            Detection(0, 0, 10, 10, 0.8, 'window'),
            Detection(0, 0, 10, 10, 0.7, 'door')
        ]
        results = DetectionResults(detections, 1.0)
        filtered = results.filter_by_class(['door'])
        
        assert len(filtered.detections) == 2
        assert all(d.class_name == 'door' for d in filtered.detections)
```

### Phase 2: Processor Tests
**Test image and PDF processing components**

```python
# tests/unit/test_image_processor.py
import pytest
import numpy as np
import cv2
from src.processors.image_processor import ImagePreprocessor

class TestImagePreprocessor:
    @pytest.fixture
    def processor(self):
        return ImagePreprocessor()
    
    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        return img
    
    @pytest.fixture
    def sample_gray_image(self):
        """Create a sample grayscale image"""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
        return img
    
    def test_convert_to_grayscale_color(self, processor, sample_color_image):
        """Test converting color image to grayscale"""
        gray = processor.convert_to_grayscale(sample_color_image)
        assert len(gray.shape) == 2
        assert gray.shape == sample_color_image.shape[:2]
    
    def test_convert_to_grayscale_already_gray(self, processor, sample_gray_image):
        """Test that grayscale images are returned unchanged"""
        gray = processor.convert_to_grayscale(sample_gray_image)
        assert np.array_equal(gray, sample_gray_image)
    
    def test_extract_edges(self, processor, sample_gray_image):
        """Test edge extraction"""
        edges = processor.extract_edges(sample_gray_image)
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8
        # Should detect edges of the rectangle
        assert np.any(edges > 0)
    
    def test_binarize_otsu(self, processor, sample_gray_image):
        """Test Otsu binarization"""
        binary = processor.binarize(sample_gray_image, method='otsu')
        assert np.all(np.isin(binary, [0, 255]))
    
    def test_resize_downscale(self, processor, sample_color_image):
        """Test image downscaling"""
        resized = processor.resize(sample_color_image, 0.5)
        assert resized.shape[0] == 50
        assert resized.shape[1] == 50
    
    def test_resize_upscale(self, processor, sample_color_image):
        """Test image upscaling"""
        resized = processor.resize(sample_color_image, 2.0)
        assert resized.shape[0] == 200
        assert resized.shape[1] == 200
```

### Phase 3: Detection Component Tests
**Test template matching and detection algorithms**

```python
# tests/unit/test_matching_engine.py
import pytest
import numpy as np
import cv2
from src.detection.matching_engine import TemplateMatchingEngine, Match

class TestTemplateMatchingEngine:
    @pytest.fixture
    def engine(self):
        return TemplateMatchingEngine('CCOEFF_NORMED')
    
    @pytest.fixture
    def scene_with_patterns(self):
        """Create scene with multiple pattern instances"""
        scene = np.ones((300, 300), dtype=np.uint8) * 255
        # Add identical rectangles
        cv2.rectangle(scene, (50, 50), (100, 100), 0, -1)
        cv2.rectangle(scene, (150, 50), (200, 100), 0, -1)
        cv2.rectangle(scene, (50, 150), (100, 200), 0, -1)
        return scene
    
    @pytest.fixture
    def template_pattern(self):
        """Create template matching the patterns"""
        template = np.ones((50, 50), dtype=np.uint8) * 255
        cv2.rectangle(template, (0, 0), (49, 49), 0, -1)
        return template
    
    def test_match_single(self, engine, scene_with_patterns, template_pattern):
        """Test single-scale matching"""
        matches = engine.match_single(
            scene_with_patterns, 
            template_pattern,
            threshold=0.8
        )
        
        assert len(matches) == 3  # Should find 3 patterns
        assert all(isinstance(m, Match) for m in matches)
        assert all(m.score > 0.8 for m in matches)
    
    def test_find_peaks(self, engine):
        """Test peak finding in response map"""
        # Create response map with clear peaks
        response = np.zeros((100, 100), dtype=np.float32)
        response[25, 25] = 0.9
        response[75, 75] = 0.85
        response[25, 75] = 0.7
        
        peaks = engine.find_peaks(response, threshold=0.6, max_peaks=10)
        
        assert len(peaks) == 3
        assert peaks[0][2] == 0.9  # Highest score first
        assert peaks[1][2] == 0.85
    
    def test_rotate_image(self, engine):
        """Test image rotation"""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (60, 60), 255, -1)
        
        rotated = engine.rotate_image(img, 45)
        
        # Rotated image should be larger to fit content
        assert rotated.shape[0] > img.shape[0]
        assert rotated.shape[1] > img.shape[1]
        assert np.any(rotated > 0)  # Should have content
    
    def test_multi_scale_match(self, engine, scene_with_patterns, template_pattern):
        """Test multi-scale matching"""
        scales = [0.9, 1.0, 1.1]
        
        matches = engine.match_multi_scale(
            scene_with_patterns,
            template_pattern,
            scales,
            threshold=0.7
        )
        
        assert len(matches) > 0
        # Should have matches at different scales
        scales_found = set(m.scale for m in matches)
        assert len(scales_found) > 1
```

### Phase 4: Integration Tests
**Test component interactions**

```python
# tests/integration/test_detection_pipeline.py
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from src.detection.detection_pipeline import DetectionPipeline
from src.models.data_models import DetectionConfig

class TestDetectionPipeline:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def test_config(self, temp_dir):
        return DetectionConfig(
            pdf_path='test.pdf',
            output_dir=temp_dir,
            class_name='test_symbol',
            threshold=0.6
        )
    
    @pytest.fixture
    def test_scene(self):
        """Create test scene with patterns"""
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # Add test patterns
        cv2.rectangle(img, (100, 100), (150, 150), (0, 0, 0), -1)
        cv2.rectangle(img, (300, 100), (350, 150), (0, 0, 0), -1)
        cv2.rectangle(img, (100, 300), (150, 350), (0, 0, 0), -1)
        return img
    
    @pytest.fixture
    def test_template(self):
        """Create test template"""
        template = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.rectangle(template, (5, 5), (45, 45), (0, 0, 0), -1)
        return template
    
    def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization"""
        pipeline = DetectionPipeline(test_config)
        
        assert pipeline.config == test_config
        assert pipeline.detector is not None
        assert Path(pipeline.output_dir).exists()
    
    def test_detector_selection(self, temp_dir):
        """Test correct detector is selected based on class name"""
        # Test HVAC detector
        config = DetectionConfig(
            pdf_path='test.pdf',
            output_dir=temp_dir,
            class_name='hvac'
        )
        pipeline = DetectionPipeline(config)
        assert 'HVAC' in pipeline.detector.__class__.__name__
        
        # Test door detector
        config.class_name = 'door'
        pipeline = DetectionPipeline(config)
        assert 'Door' in pipeline.detector.__class__.__name__
    
    def test_pipeline_run(self, test_config, test_scene, test_template):
        """Test complete pipeline execution"""
        pipeline = DetectionPipeline(test_config)
        
        results = pipeline.run(
            image=test_scene,
            template=test_template
        )
        
        assert results is not None
        assert hasattr(results, 'detections')
        assert hasattr(results, 'processing_time')
        assert results.processing_time > 0
        assert isinstance(results.detections, list)
    
    def test_save_results(self, test_config, test_scene, test_template):
        """Test result saving in multiple formats"""
        pipeline = DetectionPipeline(test_config)
        results = pipeline.run(test_scene, test_template)
        
        output_files = pipeline.save_results(results, test_scene)
        
        assert 'yolo_labels' in output_files
        assert 'json' in output_files
        assert 'summary' in output_files
        assert 'annotated_image' in output_files
        
        # Verify files exist
        for filepath in output_files.values():
            assert Path(filepath).exists()
        
        # Verify YOLO format
        with open(output_files['yolo_labels'], 'r') as f:
            lines = f.readlines()
            if results.detections:
                assert len(lines) == len(results.detections)
                # Check YOLO format (class_id cx cy w h)
                parts = lines[0].strip().split()
                assert len(parts) == 5
```

### Phase 5: End-to-End Tests
**Test complete workflows**

```python
# tests/e2e/test_full_workflow.py
import pytest
import subprocess
import tempfile
import json
from pathlib import Path

class TestEndToEndWorkflow:
    @pytest.fixture
    def sample_pdf(self):
        """Path to sample PDF for testing"""
        return "tests/fixtures/sample_pdfs/floor_plan.pdf"
    
    def test_cli_basic_detection(self, sample_pdf, tmp_path):
        """Test basic CLI detection workflow"""
        output_dir = tmp_path / "output"
        
        # Run CLI command
        cmd = [
            "python", "src/cli.py",
            "--pdf", sample_pdf,
            "--page", "0",
            "--zoom", "4.0",
            "--threshold", "0.6",
            "--class-name", "door",
            "--outdir", str(output_dir),
            "--roi", "100,100,50,50"  # Skip interactive ROI
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Detection complete" in result.stdout
        
        # Check output files
        assert (output_dir / "page0.txt").exists()  # YOLO labels
        assert (output_dir / "detections.json").exists()
        assert (output_dir / "summary.txt").exists()
    
    def test_config_save_load(self, tmp_path):
        """Test configuration save and load"""
        config_file = tmp_path / "config.json"
        
        # Save configuration
        cmd_save = [
            "python", "src/cli.py",
            "--pdf", "test.pdf",
            "--threshold", "0.7",
            "--scales", "0.9,1.0,1.1",
            "--save-config", str(config_file)
        ]
        
        subprocess.run(cmd_save)
        assert config_file.exists()
        
        # Load and verify
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        assert config['threshold'] == 0.7
        assert config['scales'] == [0.9, 1.0, 1.1]
```

## Test Execution Strategy

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/test_data_models.py

# Run tests matching pattern
pytest -k "test_detection"

# Run with verbose output
pytest -v tests/

# Run integration tests only
pytest tests/integration/
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## Test Data Management

### Fixtures Organization

```python
# tests/conftest.py
import pytest
import numpy as np
import cv2
from pathlib import Path

@pytest.fixture(scope="session")
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_floor_plan_image(fixtures_dir):
    img_path = fixtures_dir / "sample_images" / "floor_plan.png"
    return cv2.imread(str(img_path))

@pytest.fixture
def hvac_template(fixtures_dir):
    template_path = fixtures_dir / "sample_templates" / "hvac.png"
    return cv2.imread(str(template_path))

@pytest.fixture
def door_template(fixtures_dir):
    template_path = fixtures_dir / "sample_templates" / "door.png"
    return cv2.imread(str(template_path))
```

## Performance Testing

```python
# tests/performance/test_performance.py
import pytest
import time
import numpy as np
from src.detection.matching_engine import TemplateMatchingEngine

class TestPerformance:
    @pytest.mark.slow
    def test_large_image_performance(self):
        """Test performance on large images"""
        # Create large test image (4000x4000)
        large_image = np.ones((4000, 4000), dtype=np.uint8) * 255
        template = np.ones((100, 100), dtype=np.uint8) * 255
        
        engine = TemplateMatchingEngine()
        
        start = time.time()
        matches = engine.match_single(large_image, template, threshold=0.7)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.slow
    def test_multi_scale_performance(self):
        """Test multi-scale matching performance"""
        image = np.ones((2000, 2000), dtype=np.uint8) * 255
        template = np.ones((50, 50), dtype=np.uint8) * 255
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        engine = TemplateMatchingEngine()
        
        start = time.time()
        matches = engine.match_multi_scale(
            image, template, scales, threshold=0.7
        )
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # Should complete within 10 seconds
```

## Success Metrics

1. **Code Coverage**: Minimum 85% overall, 90% for critical components
2. **Test Execution Time**: Full test suite under 5 minutes
3. **Test Reliability**: Zero flaky tests
4. **Documentation**: Every test has clear docstring
5. **CI/CD Integration**: All tests pass in CI pipeline
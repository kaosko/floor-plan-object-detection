# Phase 2: PDF and Image Processing Components

## Objective
Implement PDF processing, image preprocessing, and ROI selection components.

## Tasks

### Task 2.1: PDF Processor
**File**: `src/processors/pdf_processor.py`

**Implementation**:
```python
import os
from typing import Tuple, Optional
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

class PDFProcessor:
    """Handles PDF to PNG conversion and page operations"""
    
    def __init__(self, pdf_path: str, page: int = 0, zoom: float = 6.0):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.page_index = page
        self.zoom = zoom
        self._doc = None
        
    def __enter__(self):
        self._doc = fitz.open(self.pdf_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._doc:
            self._doc.close()
    
    def render_to_png(self, output_path: str) -> str:
        """Render PDF page to PNG file"""
        with fitz.open(self.pdf_path) as doc:
            if self.page_index >= len(doc):
                raise IndexError(f"Page {self.page_index} not found in PDF")
            
            page = doc.load_page(self.page_index)
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pix.save(output_path)
            
        return output_path
    
    def render_to_array(self) -> np.ndarray:
        """Render PDF page to numpy array"""
        with fitz.open(self.pdf_path) as doc:
            page = doc.load_page(self.page_index)
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_data = pix.pil_tobytes(format="PNG")
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
    
    def get_page_count(self) -> int:
        """Get total number of pages in PDF"""
        with fitz.open(self.pdf_path) as doc:
            return len(doc)
    
    def get_page_dimensions(self, page: Optional[int] = None) -> Tuple[float, float]:
        """Get dimensions of a page"""
        if page is None:
            page = self.page_index
            
        with fitz.open(self.pdf_path) as doc:
            page_obj = doc.load_page(page)
            rect = page_obj.rect
            return rect.width * self.zoom, rect.height * self.zoom
```

**Tests**:
```python
# tests/test_pdf_processor.py
import pytest
import tempfile
import os
from src.processors.pdf_processor import PDFProcessor

@pytest.fixture
def sample_pdf():
    # Create a simple test PDF
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    page.insert_text((100, 100), "Test PDF")
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        doc.save(f.name)
        yield f.name
        os.unlink(f.name)

def test_pdf_processor_init(sample_pdf):
    processor = PDFProcessor(sample_pdf)
    assert processor.pdf_path == sample_pdf
    assert processor.page_index == 0
    assert processor.zoom == 6.0

def test_pdf_processor_render_to_png(sample_pdf):
    processor = PDFProcessor(sample_pdf, zoom=2.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'output.png')
        result = processor.render_to_png(output_path)
        
        assert os.path.exists(result)
        assert result == output_path

def test_pdf_processor_page_count(sample_pdf):
    processor = PDFProcessor(sample_pdf)
    assert processor.get_page_count() == 1

def test_pdf_processor_dimensions(sample_pdf):
    processor = PDFProcessor(sample_pdf, zoom=2.0)
    width, height = processor.get_page_dimensions()
    
    # A4 at 72 DPI with 2x zoom
    assert width == pytest.approx(595 * 2, rel=1)
    assert height == pytest.approx(842 * 2, rel=1)
```

### Task 2.2: Image Preprocessor
**File**: `src/processors/image_processor.py`

**Implementation**:
```python
import cv2
import numpy as np
from typing import Tuple, Optional

class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def extract_edges(self, image: np.ndarray, 
                     low_threshold: int = 50, 
                     high_threshold: int = 150) -> np.ndarray:
        """Extract edges using Canny edge detection"""
        gray = self.convert_to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, low_threshold, high_threshold)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        gray = self.convert_to_grayscale(image)
        return self.clahe.apply(gray)
    
    def binarize(self, image: np.ndarray, 
                 method: str = 'otsu',
                 threshold: int = 127,
                 invert: bool = True) -> np.ndarray:
        """Binarize image using various methods"""
        gray = self.convert_to_grayscale(image)
        
        if method == 'otsu':
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                11, 2
            )
        else:  # Fixed threshold
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, threshold, 255, flag)
        
        return binary
    
    def denoise(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply morphological operations to denoise"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed
    
    def resize(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Resize image by scale factor"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
```

**Tests**:
```python
# tests/test_image_processor.py
import pytest
import numpy as np
import cv2
from src.processors.image_processor import ImagePreprocessor

@pytest.fixture
def sample_image():
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    return img

def test_convert_to_grayscale(sample_image):
    processor = ImagePreprocessor()
    gray = processor.convert_to_grayscale(sample_image)
    
    assert len(gray.shape) == 2
    assert gray.shape[:2] == sample_image.shape[:2]

def test_extract_edges(sample_image):
    processor = ImagePreprocessor()
    edges = processor.extract_edges(sample_image)
    
    assert edges.shape == sample_image.shape[:2]
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)  # Should detect some edges

def test_binarize_methods(sample_image):
    processor = ImagePreprocessor()
    
    # Test Otsu method
    binary_otsu = processor.binarize(sample_image, method='otsu')
    assert binary_otsu.shape == sample_image.shape[:2]
    assert np.all(np.isin(binary_otsu, [0, 255]))
    
    # Test adaptive method
    binary_adaptive = processor.binarize(sample_image, method='adaptive')
    assert binary_adaptive.shape == sample_image.shape[:2]
    
    # Test fixed threshold
    binary_fixed = processor.binarize(sample_image, method='fixed', threshold=127)
    assert binary_fixed.shape == sample_image.shape[:2]

def test_resize(sample_image):
    processor = ImagePreprocessor()
    
    # Test downscaling
    small = processor.resize(sample_image, 0.5)
    assert small.shape[0] == 50
    assert small.shape[1] == 50
    
    # Test upscaling
    large = processor.resize(sample_image, 2.0)
    assert large.shape[0] == 200
    assert large.shape[1] == 200
```

### Task 2.3: ROI Selector
**File**: `src/processors/roi_selector.py`

**Implementation**:
```python
import cv2
import json
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

class ROISelector:
    """Handles Region of Interest selection and management"""
    
    def __init__(self):
        self.last_roi = None
        
    def select_interactive(self, image: np.ndarray, 
                          window_name: str = "Select ROI",
                          max_width: int = 1600,
                          max_height: int = 900) -> Tuple[int, int, int, int]:
        """Interactive ROI selection with scaled display"""
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        
        if scale < 1.0:
            display_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            display_img = image.copy()
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x, y, w_roi, h_roi = cv2.selectROI(window_name, display_img, 
                                           showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        
        if w_roi <= 0 or h_roi <= 0:
            raise ValueError("Invalid ROI selection")
        
        # Convert back to original image coordinates
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w_roi = int(w_roi / scale)
            h_roi = int(h_roi / scale)
        
        self.last_roi = (x, y, w_roi, h_roi)
        return self.last_roi
    
    def validate_roi(self, roi: Tuple[int, int, int, int], 
                    image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Validate and clamp ROI to image boundaries"""
        x, y, w, h = roi
        img_h, img_w = image_shape[:2]
        
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        
        return (x, y, w, h)
    
    def extract_roi(self, image: np.ndarray, 
                   roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract ROI from image"""
        x, y, w, h = self.validate_roi(roi, image.shape)
        return image[y:y+h, x:x+w].copy()
    
    def save_roi(self, roi: Tuple[int, int, int, int], filepath: str) -> None:
        """Save ROI coordinates to JSON file"""
        roi_data = {
            'x': roi[0],
            'y': roi[1],
            'width': roi[2],
            'height': roi[3]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(roi_data, f, indent=2)
    
    def load_roi(self, filepath: str) -> Tuple[int, int, int, int]:
        """Load ROI coordinates from JSON file"""
        with open(filepath, 'r') as f:
            roi_data = json.load(f)
        
        roi = (roi_data['x'], roi_data['y'], 
               roi_data['width'], roi_data['height'])
        self.last_roi = roi
        return roi
    
    def expand_roi(self, roi: Tuple[int, int, int, int], 
                  padding: int,
                  image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Expand ROI by padding amount"""
        x, y, w, h = roi
        new_roi = (x - padding, y - padding, 
                  w + 2 * padding, h + 2 * padding)
        return self.validate_roi(new_roi, image_shape)
```

**Tests**:
```python
# tests/test_roi_selector.py
import pytest
import numpy as np
import tempfile
import json
from src.processors.roi_selector import ROISelector

@pytest.fixture
def sample_image():
    return np.zeros((200, 300, 3), dtype=np.uint8)

def test_roi_validation(sample_image):
    selector = ROISelector()
    
    # Test valid ROI
    roi = selector.validate_roi((10, 10, 50, 50), sample_image.shape)
    assert roi == (10, 10, 50, 50)
    
    # Test ROI that extends beyond image
    roi = selector.validate_roi((250, 150, 100, 100), sample_image.shape)
    assert roi[0] < sample_image.shape[1]
    assert roi[1] < sample_image.shape[0]
    
    # Test negative coordinates
    roi = selector.validate_roi((-10, -10, 50, 50), sample_image.shape)
    assert roi[0] >= 0
    assert roi[1] >= 0

def test_extract_roi(sample_image):
    selector = ROISelector()
    roi = (50, 50, 100, 80)
    
    extracted = selector.extract_roi(sample_image, roi)
    assert extracted.shape == (80, 100, 3)

def test_save_load_roi():
    selector = ROISelector()
    roi = (10, 20, 100, 150)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        selector.save_roi(roi, f.name)
        loaded_roi = selector.load_roi(f.name)
        
        assert loaded_roi == roi
        assert selector.last_roi == roi

def test_expand_roi(sample_image):
    selector = ROISelector()
    roi = (50, 50, 40, 40)
    
    expanded = selector.expand_roi(roi, 10, sample_image.shape)
    assert expanded == (40, 40, 60, 60)
    
    # Test expansion at boundary
    roi = (10, 10, 20, 20)
    expanded = selector.expand_roi(roi, 20, sample_image.shape)
    assert expanded[0] == 0  # Clamped to image boundary
    assert expanded[1] == 0
```

## Deliverables
1. PDF processor with rendering capabilities
2. Image preprocessor with multiple methods
3. ROI selector with interactive and programmatic interfaces
4. All tests passing with >90% coverage
5. Documentation for each component

## Success Criteria
- PDF pages can be rendered to PNG/array
- Images can be preprocessed with various methods
- ROI can be selected interactively or loaded from file
- All edge cases handled properly
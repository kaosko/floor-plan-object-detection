# Phase 4: Symbol-Specific Detectors

## Objective
Implement base detector class and symbol-specific detectors for different architectural elements.

## Tasks

### Task 4.1: Base Symbol Detector
**File**: `src/detection/base_detector.py`

**Implementation**:
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from src.models.data_models import Detection, DetectionConfig
from src.detection.matching_engine import TemplateMatchingEngine, Match

class BaseSymbolDetector(ABC):
    """Abstract base class for all symbol detectors"""
    
    def __init__(self, class_name: str, 
                 threshold: float = 0.65,
                 config: Optional[DetectionConfig] = None):
        self.class_name = class_name
        self.threshold = threshold
        self.config = config or DetectionConfig(pdf_path='')
        self.matching_engine = TemplateMatchingEngine(self.config.method)
        self.detections = []
        
    @abstractmethod
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """Detect symbols in image using template"""
        pass
    
    @abstractmethod
    def get_detection_params(self) -> Dict:
        """Get detector-specific parameters"""
        pass
    
    def matches_to_detections(self, matches: List[Match]) -> List[Detection]:
        """Convert Match objects to Detection objects"""
        detections = []
        for match in matches:
            det = Detection(
                x1=match.x,
                y1=match.y,
                x2=match.x + match.width,
                y2=match.y + match.height,
                confidence=match.score,
                class_name=self.class_name,
                scale=match.scale,
                angle=match.angle
            )
            detections.append(det)
        return detections
    
    def apply_nms(self, detections: List[Detection], 
                 iou_threshold: float = 0.3) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicates"""
        if not detections:
            return []
        
        # Convert to numpy arrays
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Calculate areas
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def export_yolo_format(self, detections: List[Detection], 
                          image_shape: Tuple[int, int],
                          output_path: str,
                          class_id: int = 0) -> None:
        """Export detections in YOLO format"""
        h, w = image_shape[:2]
        
        with open(output_path, 'w') as f:
            for det in detections:
                cx = (det.x1 + det.x2) / 2.0
                cy = (det.y1 + det.y2) / 2.0
                bw = det.x2 - det.x1
                bh = det.y2 - det.y1
                
                # Normalize coordinates
                cx_norm = cx / w
                cy_norm = cy / h
                bw_norm = bw / w
                bh_norm = bh / h
                
                f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} "
                       f"{bw_norm:.6f} {bh_norm:.6f}\n")
    
    def filter_by_size(self, detections: List[Detection],
                      min_area: Optional[int] = None,
                      max_area: Optional[int] = None) -> List[Detection]:
        """Filter detections by bounding box area"""
        filtered = []
        for det in detections:
            area = det.area
            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue
            filtered.append(det)
        return filtered
    
    def filter_by_aspect_ratio(self, detections: List[Detection],
                              min_ratio: float = 0.5,
                              max_ratio: float = 2.0) -> List[Detection]:
        """Filter detections by aspect ratio"""
        filtered = []
        for det in detections:
            width = det.x2 - det.x1
            height = det.y2 - det.y1
            if height == 0:
                continue
            ratio = width / height
            if min_ratio <= ratio <= max_ratio:
                filtered.append(det)
        return filtered
```

**Tests**:
```python
# tests/test_base_detector.py
import pytest
import numpy as np
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection

class TestDetector(BaseSymbolDetector):
    """Concrete implementation for testing"""
    
    def detect(self, image, template):
        # Simple mock detection
        return [
            Detection(10, 10, 30, 30, 0.9, self.class_name),
            Detection(15, 15, 35, 35, 0.8, self.class_name),  # Overlapping
            Detection(100, 100, 120, 120, 0.95, self.class_name)
        ]
    
    def get_detection_params(self):
        return {'test_param': 'value'}

def test_base_detector_init():
    detector = TestDetector('test_symbol', threshold=0.7)
    assert detector.class_name == 'test_symbol'
    assert detector.threshold == 0.7

def test_apply_nms():
    detector = TestDetector('test')
    detections = detector.detect(None, None)
    
    # Apply NMS
    filtered = detector.apply_nms(detections, iou_threshold=0.3)
    
    # Should remove overlapping detection
    assert len(filtered) == 2
    assert filtered[0].confidence == 0.95  # Highest confidence first

def test_filter_by_size():
    detector = TestDetector('test')
    detections = [
        Detection(0, 0, 10, 10, 0.9, 'test'),  # area=100
        Detection(0, 0, 20, 20, 0.9, 'test'),  # area=400
        Detection(0, 0, 30, 30, 0.9, 'test'),  # area=900
    ]
    
    filtered = detector.filter_by_size(detections, min_area=150, max_area=500)
    assert len(filtered) == 1
    assert filtered[0].area == 400

def test_filter_by_aspect_ratio():
    detector = TestDetector('test')
    detections = [
        Detection(0, 0, 10, 20, 0.9, 'test'),  # ratio=0.5
        Detection(0, 0, 20, 20, 0.9, 'test'),  # ratio=1.0
        Detection(0, 0, 30, 10, 0.9, 'test'),  # ratio=3.0
    ]
    
    filtered = detector.filter_by_aspect_ratio(detections, min_ratio=0.8, max_ratio=1.2)
    assert len(filtered) == 1
    assert filtered[0].x2 == 20  # Square detection
```

### Task 4.2: HVAC Symbol Detector
**File**: `src/detection/hvac_detector.py`

**Implementation**:
```python
import numpy as np
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection

class HVACSymbolDetector(BaseSymbolDetector):
    """Detector specialized for HVAC symbols"""
    
    def __init__(self, threshold: float = 0.65):
        super().__init__("hvac_symbol", threshold)
        
        # HVAC-specific parameters
        self.min_symbol_area = 100
        self.max_symbol_area = 10000
        self.expected_aspect_ratio = 1.0  # Most HVAC symbols are roughly square
        self.aspect_ratio_tolerance = 0.3
        
    def get_detection_params(self) -> Dict:
        return {
            'min_area': self.min_symbol_area,
            'max_area': self.max_symbol_area,
            'expected_aspect_ratio': self.expected_aspect_ratio,
            'scales': [0.8, 0.9, 1.0, 1.1, 1.2],
            'angles': [0, 90, 180, 270]  # HVAC symbols often rotated
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """Detect HVAC symbols with rotation invariance"""
        
        # Use coarse-to-fine matching for efficiency
        matches = self.matching_engine.coarse_to_fine_match(
            image=image,
            template=template,
            coarse_scale=0.5,
            scales=self.get_detection_params()['scales'],
            angles=self.get_detection_params()['angles'],
            threshold=self.threshold,
            topk=200,
            refine_pad=0.3
        )
        
        # Convert to detections
        detections = self.matches_to_detections(matches)
        
        # Filter by HVAC-specific criteria
        detections = self.filter_by_size(
            detections, 
            self.min_symbol_area, 
            self.max_symbol_area
        )
        
        detections = self.filter_by_aspect_ratio(
            detections,
            self.expected_aspect_ratio - self.aspect_ratio_tolerance,
            self.expected_aspect_ratio + self.aspect_ratio_tolerance
        )
        
        # Apply NMS
        detections = self.apply_nms(detections, iou_threshold=0.25)
        
        self.detections = detections
        return detections
    
    def detect_with_context(self, image: np.ndarray,
                          template: np.ndarray,
                          context_mask: np.ndarray = None) -> List[Detection]:
        """Detect considering contextual information"""
        detections = self.detect(image, template)
        
        if context_mask is not None:
            # Filter detections based on context (e.g., only in certain areas)
            filtered = []
            for det in detections:
                cx, cy = det.center
                if context_mask[int(cy), int(cx)] > 0:
                    filtered.append(det)
            detections = filtered
        
        return detections
```

### Task 4.3: Door Symbol Detector
**File**: `src/detection/door_detector.py`

**Implementation**:
```python
import numpy as np
import cv2
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection

class DoorSymbolDetector(BaseSymbolDetector):
    """Detector specialized for door symbols"""
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("door", threshold)
        
        # Door-specific parameters
        self.min_door_width = 20
        self.max_door_width = 200
        self.door_aspect_ratios = [0.3, 2.0, 3.3]  # Various door types
        
    def get_detection_params(self) -> Dict:
        return {
            'min_width': self.min_door_width,
            'max_width': self.max_door_width,
            'scales': [0.9, 1.0, 1.1],
            'angles': [0, 90, 180, 270],  # Doors aligned with walls
            'use_line_detection': True
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """Detect door symbols with line-based validation"""
        
        # Standard template matching
        matches = self.matching_engine.match_multi_angle(
            image=image,
            template=template,
            angles=self.get_detection_params()['angles'],
            threshold=self.threshold,
            max_matches=300
        )
        
        # Convert to detections
        detections = self.matches_to_detections(matches)
        
        # Validate using line detection (doors have arc patterns)
        if self.get_detection_params()['use_line_detection']:
            detections = self._validate_with_lines(image, detections)
        
        # Apply size and aspect ratio filtering
        detections = self._filter_door_specific(detections)
        
        # Apply NMS
        detections = self.apply_nms(detections, iou_threshold=0.2)
        
        self.detections = detections
        return detections
    
    def _validate_with_lines(self, image: np.ndarray, 
                            detections: List[Detection]) -> List[Detection]:
        """Validate detections using line/arc detection"""
        validated = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        for det in detections:
            # Extract region
            roi = gray[det.y1:det.y2, det.x1:det.x2]
            
            if roi.size == 0:
                continue
            
            # Detect edges
            edges = cv2.Canny(roi, 50, 150)
            
            # Check for arc patterns (simplified)
            # Doors typically have curved lines
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            # If arc/circle detected, likely a door
            if circles is not None or np.sum(edges) > 100:
                validated.append(det)
        
        return validated
    
    def _filter_door_specific(self, detections: List[Detection]) -> List[Detection]:
        """Apply door-specific filtering"""
        filtered = []
        
        for det in detections:
            width = det.x2 - det.x1
            
            # Check width constraints
            if not (self.min_door_width <= width <= self.max_door_width):
                continue
            
            # Check if aspect ratio matches any door type
            height = det.y2 - det.y1
            if height == 0:
                continue
                
            aspect_ratio = width / height
            
            # Allow some tolerance around expected ratios
            valid_ratio = any(
                abs(aspect_ratio - expected) < 0.5 
                for expected in self.door_aspect_ratios
            )
            
            if valid_ratio:
                filtered.append(det)
        
        return filtered
```

### Task 4.4: Window Symbol Detector
**File**: `src/detection/window_detector.py`

**Implementation**:
```python
import numpy as np
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection

class WindowSymbolDetector(BaseSymbolDetector):
    """Detector specialized for window symbols"""
    
    def __init__(self, threshold: float = 0.68):
        super().__init__("window", threshold)
        
        # Window-specific parameters
        self.parallel_line_threshold = 0.8
        self.min_window_length = 30
        self.max_window_length = 300
        
    def get_detection_params(self) -> Dict:
        return {
            'min_length': self.min_window_length,
            'max_length': self.max_window_length,
            'scales': [0.95, 1.0, 1.05],
            'angles': [0, 90],  # Windows typically horizontal or vertical
            'detect_parallel_lines': True
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """Detect window symbols (typically parallel lines)"""
        
        # Windows are often simpler patterns
        matches = self.matching_engine.match_multi_scale(
            image=image,
            template=template,
            scales=self.get_detection_params()['scales'],
            threshold=self.threshold,
            max_matches=200
        )
        
        # Convert to detections
        detections = self.matches_to_detections(matches)
        
        # Validate window patterns
        if self.get_detection_params()['detect_parallel_lines']:
            detections = self._validate_window_pattern(image, detections)
        
        # Filter by size
        detections = self._filter_by_window_dimensions(detections)
        
        # Apply NMS with lower threshold (windows can be close)
        detections = self.apply_nms(detections, iou_threshold=0.15)
        
        self.detections = detections
        return detections
    
    def _validate_window_pattern(self, image: np.ndarray,
                                detections: List[Detection]) -> List[Detection]:
        """Validate that detection contains window pattern"""
        # Simplified validation - in practice would check for parallel lines
        validated = []
        
        for det in detections:
            # Windows typically have good edge response
            if det.confidence > self.parallel_line_threshold:
                validated.append(det)
        
        return validated
    
    def _filter_by_window_dimensions(self, 
                                    detections: List[Detection]) -> List[Detection]:
        """Filter based on window-specific dimensions"""
        filtered = []
        
        for det in detections:
            length = max(det.x2 - det.x1, det.y2 - det.y1)
            
            if self.min_window_length <= length <= self.max_window_length:
                filtered.append(det)
        
        return filtered
    
    def detect_window_groups(self, image: np.ndarray,
                           template: np.ndarray,
                           group_threshold: float = 50) -> List[List[Detection]]:
        """Detect and group aligned windows"""
        detections = self.detect(image, template)
        
        # Group windows that are aligned
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check if aligned horizontally or vertically
                h_aligned = abs(det1.center[1] - det2.center[1]) < group_threshold
                v_aligned = abs(det1.center[0] - det2.center[0]) < group_threshold
                
                if h_aligned or v_aligned:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
```

**Tests**:
```python
# tests/test_symbol_detectors.py
import pytest
import numpy as np
import cv2
from src.detection.hvac_detector import HVACSymbolDetector
from src.detection.door_detector import DoorSymbolDetector
from src.detection.window_detector import WindowSymbolDetector

@pytest.fixture
def sample_image():
    # Create test image with patterns
    img = np.ones((300, 300), dtype=np.uint8) * 255
    # Add some rectangular patterns
    cv2.rectangle(img, (50, 50), (100, 100), 0, -1)
    cv2.rectangle(img, (150, 150), (200, 200), 0, -1)
    return img

@pytest.fixture
def hvac_template():
    # Square template for HVAC
    template = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.rectangle(template, (10, 10), (40, 40), 0, -1)
    return template

def test_hvac_detector(sample_image, hvac_template):
    detector = HVACSymbolDetector(threshold=0.6)
    params = detector.get_detection_params()
    
    assert 'scales' in params
    assert 'angles' in params
    assert len(params['angles']) == 4  # 4 rotations
    
    # Test detection (would need proper test data)
    detections = detector.detect(sample_image, hvac_template)
    assert isinstance(detections, list)

def test_door_detector():
    detector = DoorSymbolDetector(threshold=0.7)
    params = detector.get_detection_params()
    
    assert params['use_line_detection'] == True
    assert 'angles' in params
    
    # Test door-specific filtering
    from src.models.data_models import Detection
    test_detections = [
        Detection(0, 0, 30, 100, 0.9, 'door'),  # Valid door ratio
        Detection(0, 0, 10, 10, 0.9, 'door'),   # Too small
    ]
    
    filtered = detector._filter_door_specific(test_detections)
    assert len(filtered) == 1

def test_window_detector():
    detector = WindowSymbolDetector(threshold=0.68)
    params = detector.get_detection_params()
    
    assert params['detect_parallel_lines'] == True
    assert len(params['angles']) == 2  # Horizontal and vertical
    
    # Test window grouping
    from src.models.data_models import Detection
    detections = [
        Detection(10, 50, 40, 70, 0.9, 'window'),
        Detection(60, 50, 90, 70, 0.9, 'window'),  # Aligned horizontally
        Detection(10, 150, 40, 170, 0.9, 'window'),  # Different row
    ]
    
    detector.detections = detections
    groups = detector.detect_window_groups(None, None, group_threshold=30)
    
    assert len(groups) == 2  # Two groups
    assert len(groups[0]) == 2  # First two windows grouped
```

## Deliverables
1. Base detector class with common functionality
2. HVAC symbol detector with rotation handling
3. Door detector with arc validation
4. Window detector with grouping capability
5. All tests passing with >85% coverage
6. Documentation for each detector's approach

## Success Criteria
- Each detector properly inherits from base class
- Symbol-specific logic is correctly implemented
- Detectors can handle various orientations and scales
- NMS and filtering work correctly
- Export to YOLO format functions properly
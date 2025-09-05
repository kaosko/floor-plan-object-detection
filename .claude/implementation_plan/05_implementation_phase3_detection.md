# Phase 3: Template Matching and Detection Components

## Objective
Implement template extraction, matching engine, and symbol-specific detectors.

## Tasks

### Task 3.1: Template Extractor
**File**: `src/detection/template_extractor.py`

**Implementation**:
```python
import cv2
import numpy as np
from typing import Tuple, Dict, Optional

class TemplateExtractor:
    """Extracts and analyzes templates from ROI"""
    
    def __init__(self):
        self.last_template = None
        self.last_mask = None
        
    def extract_template(self, image: np.ndarray, 
                        roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract template from ROI"""
        x, y, w, h = roi
        template = image[y:y+h, x:x+w].copy()
        self.last_template = template
        return template
    
    def create_interior_mask(self, template: np.ndarray,
                           use_otsu: bool = True,
                           thresh_val: int = 127,
                           invert: bool = True,
                           close_ksize: int = 3,
                           min_area_ratio: float = 0.003,
                           keep_mode: str = "all") -> np.ndarray:
        """Create mask of interior regions (holes) in symbol"""
        # Convert to grayscale if needed
        if len(template.shape) == 3:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template.copy()
        
        # Binarize
        if use_otsu:
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        else:
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, thresh_val, 255, flag)
        
        # Morphological closing to seal gaps
        if close_ksize > 0:
            kernel = np.ones((close_ksize, close_ksize), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        h, w = gray.shape
        mask = np.zeros((h, w), np.uint8)
        
        if hierarchy is None or len(contours) == 0:
            self.last_mask = mask
            return mask
        
        # Process holes (contours with parent)
        roi_area = float(h * w)
        keep_indices = []
        
        for i, cnt in enumerate(contours):
            parent = hierarchy[0][i][3]
            if parent == -1:  # Skip outer contours
                continue
                
            area = cv2.contourArea(cnt)
            if area < min_area_ratio * roi_area:
                continue
            
            if keep_mode == "largest":
                keep_indices.append((i, area))
            else:
                keep_indices.append((i, area))
        
        # Draw selected contours
        if keep_mode == "largest" and keep_indices:
            keep_indices.sort(key=lambda x: x[1], reverse=True)
            cv2.drawContours(mask, contours, keep_indices[0][0], 255, -1)
        else:
            for idx, _ in keep_indices:
                cv2.drawContours(mask, contours, idx, 255, -1)
        
        # Final cleanup
        if close_ksize > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.last_mask = mask
        return mask
    
    def tighten_roi(self, roi: Tuple[int, int, int, int],
                   mask: np.ndarray,
                   image_shape: Tuple[int, int],
                   pad_px: int = 2,
                   pad_ratio: float = 0.03) -> Tuple[int, int, int, int]:
        """Tighten ROI based on mask bounds"""
        if mask.ndim == 3:
            mask = mask[..., 0]
        
        # Find mask bounds
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return roi
        
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        
        # Calculate padding
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        pad = int(round(pad_px + pad_ratio * max(bw, bh)))
        
        # Map to image coordinates and clamp
        x, y, w, h = roi
        h_img, w_img = image_shape
        
        nx = max(0, x + x0 - pad)
        ny = max(0, y + y0 - pad)
        nx2 = min(w_img - 1, x + x1 + pad)
        ny2 = min(h_img - 1, y + y1 + pad)
        
        nw = max(8, nx2 - nx + 1)
        nh = max(8, ny2 - ny + 1)
        
        return (nx, ny, nw, nh)
    
    def analyze_template_features(self, template: np.ndarray) -> Dict:
        """Analyze template characteristics"""
        features = {}
        
        # Convert to grayscale
        if len(template.shape) == 3:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template
        
        # Basic statistics
        features['mean'] = float(np.mean(gray))
        features['std'] = float(np.std(gray))
        features['min'] = float(np.min(gray))
        features['max'] = float(np.max(gray))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0)) / edges.size
        
        # Contour properties
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            features['largest_contour_area'] = float(cv2.contourArea(largest))
            features['num_contours'] = len(contours)
        else:
            features['largest_contour_area'] = 0.0
            features['num_contours'] = 0
        
        return features
```

**Tests**:
```python
# tests/test_template_extractor.py
import pytest
import numpy as np
import cv2
from src.detection.template_extractor import TemplateExtractor

@pytest.fixture
def sample_image_with_symbol():
    # Create image with a simple symbol (square with hole)
    img = np.ones((100, 100), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), 0, 2)  # Outer rectangle
    cv2.rectangle(img, (40, 40), (60, 60), 255, -1)  # Inner hole
    return img

def test_extract_template(sample_image_with_symbol):
    extractor = TemplateExtractor()
    roi = (10, 10, 80, 80)
    
    template = extractor.extract_template(sample_image_with_symbol, roi)
    assert template.shape == (80, 80)
    assert extractor.last_template is not None

def test_create_interior_mask(sample_image_with_symbol):
    extractor = TemplateExtractor()
    roi = (10, 10, 80, 80)
    template = extractor.extract_template(sample_image_with_symbol, roi)
    
    mask = extractor.create_interior_mask(template)
    assert mask.shape == template.shape
    assert np.any(mask > 0)  # Should detect interior regions

def test_analyze_template_features(sample_image_with_symbol):
    extractor = TemplateExtractor()
    features = extractor.analyze_template_features(sample_image_with_symbol)
    
    assert 'mean' in features
    assert 'edge_density' in features
    assert 'num_contours' in features
    assert features['num_contours'] >= 1
```

### Task 3.2: Template Matching Engine
**File**: `src/detection/matching_engine.py`

**Implementation**:
```python
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Match:
    """Represents a template match"""
    x: int
    y: int
    width: int
    height: int
    score: float
    scale: float = 1.0
    angle: float = 0.0

class TemplateMatchingEngine:
    """Core template matching operations"""
    
    def __init__(self, method: str = 'CCOEFF_NORMED'):
        self.method_map = {
            'CCOEFF': cv2.TM_CCOEFF,
            'CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
            'CCORR_NORMED': cv2.TM_CCORR_NORMED,
            'SQDIFF': cv2.TM_SQDIFF,
            'SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
        }
        self.method = self.method_map.get(method, cv2.TM_CCOEFF_NORMED)
        self.is_sqdiff = method in ['SQDIFF', 'SQDIFF_NORMED']
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image keeping all content"""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        
        # Calculate new dimensions
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        
        return cv2.warpAffine(image, M, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    
    def find_peaks(self, response: np.ndarray, 
                  threshold: float,
                  max_peaks: int = 100) -> List[Tuple[int, int, float]]:
        """Find local maxima in response map"""
        if response.size == 0:
            return []
        
        # For SQDIFF methods, invert the response
        if self.is_sqdiff:
            response = 1.0 - response
            
        # Find local maxima using dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(response, kernel)
        
        # Peaks are points where original equals dilated and above threshold
        peaks_mask = (response == dilated) & (response >= threshold)
        
        # Get peak coordinates and scores
        ys, xs = np.where(peaks_mask)
        scores = response[peaks_mask]
        
        # Sort by score and keep top peaks
        peaks = [(int(x), int(y), float(s)) for x, y, s in zip(xs, ys, scores)]
        peaks.sort(key=lambda p: p[2], reverse=True)
        
        return peaks[:max_peaks]
    
    def match_single(self, image: np.ndarray, 
                    template: np.ndarray,
                    threshold: float = 0.7) -> List[Match]:
        """Single scale/angle template matching"""
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            return []
        
        # Perform template matching
        response = cv2.matchTemplate(image, template, self.method)
        
        # Find peaks
        peaks = self.find_peaks(response, threshold)
        
        # Convert to Match objects
        matches = []
        th, tw = template.shape[:2]
        for x, y, score in peaks:
            matches.append(Match(x, y, tw, th, score))
        
        return matches
    
    def match_multi_scale(self, image: np.ndarray,
                         template: np.ndarray,
                         scales: List[float],
                         threshold: float = 0.7,
                         max_matches: int = 100) -> List[Match]:
        """Multi-scale template matching"""
        all_matches = []
        
        for scale in scales:
            # Resize template
            th, tw = template.shape[:2]
            new_w = max(8, int(tw * scale))
            new_h = max(8, int(th * scale))
            
            if new_w >= image.shape[1] or new_h >= image.shape[0]:
                continue
            
            scaled_template = cv2.resize(template, (new_w, new_h), 
                                        interpolation=cv2.INTER_AREA)
            
            # Find matches at this scale
            matches = self.match_single(image, scaled_template, threshold)
            
            # Update scale information
            for match in matches:
                match.scale = scale
            
            all_matches.extend(matches)
        
        # Sort by score and limit
        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:max_matches]
    
    def match_multi_angle(self, image: np.ndarray,
                         template: np.ndarray,
                         angles: List[float],
                         threshold: float = 0.7,
                         max_matches: int = 100) -> List[Match]:
        """Multi-angle template matching"""
        all_matches = []
        
        for angle in angles:
            # Rotate template if needed
            if abs(angle) > 1e-3:
                rotated = self.rotate_image(template, angle)
            else:
                rotated = template
            
            # Find matches at this angle
            matches = self.match_single(image, rotated, threshold)
            
            # Update angle information
            for match in matches:
                match.angle = angle
            
            all_matches.extend(matches)
        
        # Sort by score and limit
        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:max_matches]
    
    def coarse_to_fine_match(self, image: np.ndarray,
                            template: np.ndarray,
                            coarse_scale: float = 0.5,
                            scales: List[float] = [1.0],
                            angles: List[float] = [0.0],
                            threshold: float = 0.7,
                            topk: int = 100,
                            refine_pad: float = 0.5) -> List[Match]:
        """Two-stage coarse-to-fine matching"""
        
        # Stage 1: Coarse search on downscaled image
        if coarse_scale < 1.0:
            small_image = cv2.resize(image, 
                                    (int(image.shape[1] * coarse_scale),
                                     int(image.shape[0] * coarse_scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            small_image = image
        
        coarse_matches = []
        
        for angle in angles:
            if abs(angle) > 1e-3:
                tmpl_rot = self.rotate_image(template, angle)
            else:
                tmpl_rot = template
            
            for scale in scales:
                # Scale template for coarse search
                th, tw = tmpl_rot.shape[:2]
                coarse_w = max(8, int(tw * scale * coarse_scale))
                coarse_h = max(8, int(th * scale * coarse_scale))
                
                if coarse_w >= small_image.shape[1] or coarse_h >= small_image.shape[0]:
                    continue
                
                coarse_tmpl = cv2.resize(tmpl_rot, (coarse_w, coarse_h),
                                        interpolation=cv2.INTER_AREA)
                
                # Match at coarse level
                matches = self.match_single(small_image, coarse_tmpl, threshold)
                
                # Convert to full resolution coordinates
                for match in matches:
                    match.x = int(match.x / coarse_scale)
                    match.y = int(match.y / coarse_scale)
                    match.width = int(match.width / coarse_scale)
                    match.height = int(match.height / coarse_scale)
                    match.scale = scale
                    match.angle = angle
                
                coarse_matches.extend(matches)
        
        # Sort and keep top K
        coarse_matches.sort(key=lambda m: m.score, reverse=True)
        coarse_matches = coarse_matches[:topk]
        
        # Stage 2: Refine around coarse matches
        refined_matches = []
        
        for coarse in coarse_matches:
            # Prepare template at detected scale/angle
            if abs(coarse.angle) > 1e-3:
                tmpl_rot = self.rotate_image(template, coarse.angle)
            else:
                tmpl_rot = template
            
            th, tw = tmpl_rot.shape[:2]
            fine_w = max(8, int(tw * coarse.scale))
            fine_h = max(8, int(th * coarse.scale))
            
            fine_tmpl = cv2.resize(tmpl_rot, (fine_w, fine_h),
                                  interpolation=cv2.INTER_AREA)
            
            # Define refinement region
            pad_w = int(fine_w * refine_pad)
            pad_h = int(fine_h * refine_pad)
            
            rx1 = max(0, coarse.x - pad_w)
            ry1 = max(0, coarse.y - pad_h)
            rx2 = min(image.shape[1], coarse.x + fine_w + pad_w)
            ry2 = min(image.shape[0], coarse.y + fine_h + pad_h)
            
            roi = image[ry1:ry2, rx1:rx2]
            
            if roi.shape[0] < fine_h or roi.shape[1] < fine_w:
                refined_matches.append(coarse)
                continue
            
            # Refine within ROI
            fine_matches = self.match_single(roi, fine_tmpl, threshold * 0.95)
            
            if fine_matches:
                best = max(fine_matches, key=lambda m: m.score)
                best.x += rx1
                best.y += ry1
                best.scale = coarse.scale
                best.angle = coarse.angle
                refined_matches.append(best)
            else:
                refined_matches.append(coarse)
        
        return refined_matches
```

**Tests**:
```python
# tests/test_matching_engine.py
import pytest
import numpy as np
import cv2
from src.detection.matching_engine import TemplateMatchingEngine, Match

@pytest.fixture
def sample_scene():
    # Create scene with multiple instances of a pattern
    scene = np.ones((200, 200), dtype=np.uint8) * 255
    # Add rectangles at different positions
    cv2.rectangle(scene, (20, 20), (40, 40), 0, -1)
    cv2.rectangle(scene, (100, 50), (120, 70), 0, -1)
    cv2.rectangle(scene, (150, 150), (170, 170), 0, -1)
    return scene

@pytest.fixture
def sample_template():
    # Create template (small rectangle)
    template = np.ones((20, 20), dtype=np.uint8) * 255
    cv2.rectangle(template, (0, 0), (19, 19), 0, -1)
    return template

def test_match_single(sample_scene, sample_template):
    engine = TemplateMatchingEngine()
    matches = engine.match_single(sample_scene, sample_template, threshold=0.9)
    
    assert len(matches) == 3  # Should find 3 rectangles
    assert all(isinstance(m, Match) for m in matches)
    assert all(m.score > 0.9 for m in matches)

def test_match_multi_scale(sample_scene, sample_template):
    engine = TemplateMatchingEngine()
    scales = [0.8, 1.0, 1.2]
    
    matches = engine.match_multi_scale(
        sample_scene, sample_template, scales, threshold=0.8
    )
    
    assert len(matches) > 0
    assert any(m.scale != 1.0 for m in matches)  # Should have different scales

def test_rotate_image():
    engine = TemplateMatchingEngine()
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (60, 60), 255, -1)
    
    rotated = engine.rotate_image(img, 45)
    assert rotated.shape[0] > img.shape[0]  # Should be larger to fit rotated content
    assert np.any(rotated > 0)  # Should have non-zero pixels

def test_coarse_to_fine(sample_scene, sample_template):
    engine = TemplateMatchingEngine()
    
    matches = engine.coarse_to_fine_match(
        sample_scene, sample_template,
        coarse_scale=0.5,
        scales=[0.9, 1.0, 1.1],
        threshold=0.8
    )
    
    assert len(matches) > 0
    assert all(isinstance(m, Match) for m in matches)
```

## Deliverables
1. Template extractor with mask generation
2. Template matching engine with multi-scale/angle support
3. Coarse-to-fine matching implementation
4. All tests passing with >90% coverage
5. Documentation for matching strategies

## Success Criteria
- Templates can be extracted and analyzed
- Multi-scale and multi-angle matching works correctly
- Coarse-to-fine strategy improves performance
- All matching methods properly tested
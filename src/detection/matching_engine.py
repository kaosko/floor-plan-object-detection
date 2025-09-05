import cv2
import numpy as np
from typing import List, Tuple, Dict
from src.models.data_models import Detection

class TemplateMatchingEngine:
    """Core template matching operations"""
    
    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED):
        self.method = method
    
    def coarse_to_fine_match(self, image: np.ndarray, template: np.ndarray,
                           coarse_scale: float, scales: List[float], angles: List[float],
                           threshold: float, topk: int, refine_pad: float) -> List[Detection]:
        """
        Perform coarse-to-fine template matching
        
        Args:
            image: Search image
            template: Template to match
            coarse_scale: Scale for coarse search (0-1)
            scales: List of scales for fine search
            angles: List of angles for fine search
            threshold: Match threshold
            topk: Number of top candidates to keep from coarse search
            refine_pad: Padding ratio for refinement
            
        Returns:
            List of detections
        """
        detections = []
        
        # Stage 1: Coarse search
        coarse_candidates = self._coarse_search(
            image, template, coarse_scale, scales, angles, topk
        )
        
        # Stage 2: Fine refinement
        for candidate in coarse_candidates:
            refined = self._refine_candidate(
                image, template, candidate, scales, angles, threshold, refine_pad
            )
            if refined and refined.confidence >= threshold:
                detections.append(refined)
        
        return detections
    
    def _coarse_search(self, image: np.ndarray, template: np.ndarray,
                      coarse_scale: float, scales: List[float], angles: List[float],
                      topk: int) -> List[Dict]:
        """Perform coarse search at reduced resolution"""
        # Resize image and template for coarse search
        coarse_h, coarse_w = int(image.shape[0] * coarse_scale), int(image.shape[1] * coarse_scale)
        coarse_image = cv2.resize(image, (coarse_w, coarse_h))
        
        candidates = []
        
        for scale in scales:
            for angle in angles:
                # Transform template
                transformed_template = self._transform_template(template, scale, angle)
                
                if transformed_template.shape[0] >= coarse_image.shape[0] or \
                   transformed_template.shape[1] >= coarse_image.shape[1]:
                    continue
                
                # Resize template for coarse search
                template_h, template_w = transformed_template.shape[:2]
                coarse_template_h = int(template_h * coarse_scale)
                coarse_template_w = int(template_w * coarse_scale)
                
                if coarse_template_h <= 0 or coarse_template_w <= 0:
                    continue
                    
                coarse_template = cv2.resize(transformed_template, (coarse_template_w, coarse_template_h))
                
                # Perform template matching
                result = cv2.matchTemplate(coarse_image, coarse_template, self.method)
                
                # Find local maxima
                locations = self._find_local_maxima(result, min_distance=10)
                
                for y, x in locations:
                    confidence = result[y, x]
                    
                    # Convert back to original coordinates
                    orig_x = int(x / coarse_scale)
                    orig_y = int(y / coarse_scale)
                    
                    candidates.append({
                        'x': orig_x,
                        'y': orig_y,
                        'scale': scale,
                        'angle': angle,
                        'confidence': confidence,
                        'template_shape': (template_h, template_w)
                    })
        
        # Sort by confidence and keep top-k
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates[:topk]
    
    def _refine_candidate(self, image: np.ndarray, template: np.ndarray,
                         candidate: Dict, scales: List[float], angles: List[float],
                         threshold: float, refine_pad: float) -> Detection:
        """Refine a candidate detection at full resolution"""
        best_detection = None
        best_confidence = 0
        
        # Extract region around candidate for refinement
        x, y = candidate['x'], candidate['y']
        template_h, template_w = candidate['template_shape']
        
        # Calculate refinement region with padding
        pad_w = int(template_w * refine_pad)
        pad_h = int(template_h * refine_pad)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + template_w + pad_w)
        y2 = min(image.shape[0], y + template_h + pad_h)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Try different scales and angles around the coarse candidate
        search_scales = [candidate['scale']]
        search_angles = [candidate['angle']]
        
        for scale in search_scales:
            for angle in search_angles:
                transformed_template = self._transform_template(template, scale, angle)
                
                if transformed_template.shape[0] >= roi.shape[0] or \
                   transformed_template.shape[1] >= roi.shape[1]:
                    continue
                
                result = cv2.matchTemplate(roi, transformed_template, self.method)
                
                if result.size == 0:
                    continue
                
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    
                    # Convert coordinates back to original image
                    roi_x, roi_y = max_loc
                    orig_x = x1 + roi_x
                    orig_y = y1 + roi_y
                    orig_w, orig_h = transformed_template.shape[1], transformed_template.shape[0]
                    
                    best_detection = Detection(
                        x1=orig_x,
                        y1=orig_y,
                        x2=orig_x + orig_w,
                        y2=orig_y + orig_h,
                        confidence=best_confidence,
                        class_name='symbol',  # Will be set by detector
                        scale=scale,
                        angle=angle
                    )
        
        return best_detection
    
    def _transform_template(self, template: np.ndarray, scale: float, angle: float) -> np.ndarray:
        """Apply scale and rotation transformations to template"""
        if scale != 1.0:
            new_h, new_w = int(template.shape[0] * scale), int(template.shape[1] * scale)
            if new_h <= 0 or new_w <= 0:
                return template
            template = cv2.resize(template, (new_w, new_h))
        
        if angle != 0:
            center = (template.shape[1] // 2, template.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new bounding dimensions
            cos_theta = abs(rotation_matrix[0, 0])
            sin_theta = abs(rotation_matrix[0, 1])
            
            new_w = int((template.shape[0] * sin_theta) + (template.shape[1] * cos_theta))
            new_h = int((template.shape[0] * cos_theta) + (template.shape[1] * sin_theta))
            
            # Adjust the rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            template = cv2.warpAffine(template, rotation_matrix, (new_w, new_h))
        
        return template
    
    def _find_local_maxima(self, image: np.ndarray, min_distance: int = 10) -> List[Tuple[int, int]]:
        """Find local maxima in response map"""
        # Simple peak detection - could be enhanced with more sophisticated methods
        kernel = np.ones((min_distance, min_distance))
        local_max = cv2.dilate(image, kernel)
        
        peaks = []
        locations = np.where((image == local_max) & (image > 0))
        
        for y, x in zip(locations[0], locations[1]):
            peaks.append((y, x))
        
        return peaks
    
    def multi_scale_match(self, image: np.ndarray, template: np.ndarray, scales: List[float]) -> List[Detection]:
        """Perform multi-scale template matching"""
        detections = []
        
        for scale in scales:
            scaled_template = self._transform_template(template, scale, 0)
            
            if scaled_template.shape[0] >= image.shape[0] or scaled_template.shape[1] >= image.shape[1]:
                continue
            
            result = cv2.matchTemplate(image, scaled_template, self.method)
            locations = np.where(result >= 0.5)  # Threshold will be applied later
            
            for y, x in zip(locations[0], locations[1]):
                confidence = result[y, x]
                w, h = scaled_template.shape[1], scaled_template.shape[0]
                
                detection = Detection(
                    x1=x, y1=y, x2=x+w, y2=y+h,
                    confidence=confidence,
                    class_name='symbol',
                    scale=scale,
                    angle=0
                )
                detections.append(detection)
        
        return detections
    
    def multi_angle_match(self, image: np.ndarray, template: np.ndarray, angles: List[float]) -> List[Detection]:
        """Perform multi-angle template matching"""
        detections = []
        
        for angle in angles:
            rotated_template = self._transform_template(template, 1.0, angle)
            
            if rotated_template.shape[0] >= image.shape[0] or rotated_template.shape[1] >= image.shape[1]:
                continue
            
            result = cv2.matchTemplate(image, rotated_template, self.method)
            locations = np.where(result >= 0.5)  # Threshold will be applied later
            
            for y, x in zip(locations[0], locations[1]):
                confidence = result[y, x]
                w, h = rotated_template.shape[1], rotated_template.shape[0]
                
                detection = Detection(
                    x1=x, y1=y, x2=x+w, y2=y+h,
                    confidence=confidence,
                    class_name='symbol',
                    scale=1.0,
                    angle=angle
                )
                detections.append(detection)
        
        return detections
"""
HVAC Symbol Detector with rotation handling and specialized filtering.
"""
import cv2
import numpy as np
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection


class HVACSymbolDetector(BaseSymbolDetector):
    """Detector specialized for HVAC symbols"""
    
    def __init__(self, threshold: float = 0.65):
        """
        Initialize HVAC detector.
        
        Args:
            threshold: Detection confidence threshold
        """
        super().__init__("hvac_symbol", threshold)
        
        # HVAC-specific parameters
        self.min_symbol_area = 100
        self.max_symbol_area = 10000
        self.expected_aspect_ratio = 1.0  # Most HVAC symbols are roughly square
        self.aspect_ratio_tolerance = 0.3
        
    def get_detection_params(self) -> Dict:
        """
        Get HVAC-specific detection parameters.
        
        Returns:
            Dictionary of detection parameters
        """
        return {
            'min_area': self.min_symbol_area,
            'max_area': self.max_symbol_area,
            'expected_aspect_ratio': self.expected_aspect_ratio,
            'scales': [0.8, 0.9, 1.0, 1.1, 1.2],
            'angles': [0, 90, 180, 270]  # HVAC symbols often rotated
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """
        Detect HVAC symbols with rotation invariance.
        
        Args:
            image: Target image
            template: Template image
            
        Returns:
            List of HVAC symbol detections
        """
        
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
        
        # Additional HVAC-specific validation
        detections = self._validate_hvac_symbols(image, detections)
        
        # Apply NMS
        detections = self.apply_nms(detections, iou_threshold=0.25)
        
        self.detections = detections
        return detections
    
    def detect_with_context(self, image: np.ndarray,
                          template: np.ndarray,
                          context_mask: np.ndarray = None) -> List[Detection]:
        """
        Detect considering contextual information.
        
        Args:
            image: Target image
            template: Template image
            context_mask: Binary mask indicating valid regions
            
        Returns:
            List of contextually filtered detections
        """
        detections = self.detect(image, template)
        
        if context_mask is not None:
            # Filter detections based on context (e.g., only in certain areas)
            filtered = []
            for det in detections:
                cx, cy = det.center
                cx, cy = int(cx), int(cy)
                if (0 <= cx < context_mask.shape[1] and 
                    0 <= cy < context_mask.shape[0] and
                    context_mask[cy, cx] > 0):
                    filtered.append(det)
            detections = filtered
        
        return detections
    
    def _validate_hvac_symbols(self, image: np.ndarray, 
                             detections: List[Detection]) -> List[Detection]:
        """
        Validate detections using HVAC-specific criteria.
        
        Args:
            image: Source image
            detections: List of detections to validate
            
        Returns:
            Validated detections
        """
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
            
            # HVAC symbols typically have distinct shape characteristics
            if self._has_hvac_characteristics(roi):
                validated.append(det)
        
        return validated
    
    def _has_hvac_characteristics(self, roi: np.ndarray) -> bool:
        """
        Check if ROI has characteristics typical of HVAC symbols.
        
        Args:
            roi: Region of interest
            
        Returns:
            True if ROI has HVAC characteristics
        """
        if roi.size == 0:
            return False
        
        # Check for circular/elliptical patterns (common in HVAC)
        # Find contours
        try:
            import cv2
            
            # Apply threshold
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # Analyze largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < 50:  # Too small
                return False
            
            # Check circularity
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter == 0:
                return False
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # HVAC symbols often have circular components
            if circularity > 0.3:  # Reasonably circular
                return True
            
            # Check for rectangular patterns (also common)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h
            
            # Check if roughly square (common for HVAC)
            if 0.7 <= aspect_ratio <= 1.3:
                return True
            
        except Exception:
            # If opencv operations fail, use simpler heuristics
            pass
        
        # Simple edge-based validation
        edges = np.abs(np.gradient(roi.astype(float)))
        edge_density = np.mean(edges[0]**2 + edges[1]**2)
        
        # HVAC symbols should have moderate edge density
        return 0.1 < edge_density < 100.0
    
    def detect_by_type(self, image: np.ndarray, 
                      hvac_type: str = 'general') -> List[Detection]:
        """
        Detect specific types of HVAC symbols.
        
        Args:
            image: Target image
            hvac_type: Type of HVAC symbol ('vent', 'duct', 'unit', 'general')
            
        Returns:
            List of type-specific detections
        """
        # This would require different templates for different HVAC types
        # For now, use general detection with type-specific parameters
        
        type_params = {
            'vent': {
                'min_area': 50,
                'max_area': 2000,
                'expected_ratio': 1.0,
                'tolerance': 0.5
            },
            'duct': {
                'min_area': 200,
                'max_area': 5000,
                'expected_ratio': 2.0,  # Often rectangular
                'tolerance': 1.0
            },
            'unit': {
                'min_area': 1000,
                'max_area': 20000,
                'expected_ratio': 1.2,
                'tolerance': 0.8
            },
            'general': {
                'min_area': self.min_symbol_area,
                'max_area': self.max_symbol_area,
                'expected_ratio': self.expected_aspect_ratio,
                'tolerance': self.aspect_ratio_tolerance
            }
        }
        
        params = type_params.get(hvac_type, type_params['general'])
        
        # Temporarily adjust parameters
        orig_min_area = self.min_symbol_area
        orig_max_area = self.max_symbol_area
        orig_ratio = self.expected_aspect_ratio
        orig_tolerance = self.aspect_ratio_tolerance
        
        self.min_symbol_area = params['min_area']
        self.max_symbol_area = params['max_area']
        self.expected_aspect_ratio = params['expected_ratio']
        self.aspect_ratio_tolerance = params['tolerance']
        
        # Note: This would need a template parameter, but for demonstration
        # we'll use a mock template
        mock_template = np.ones((50, 50), dtype=np.uint8) * 255
        detections = self.detect(image, mock_template)
        
        # Restore original parameters
        self.min_symbol_area = orig_min_area
        self.max_symbol_area = orig_max_area
        self.expected_aspect_ratio = orig_ratio
        self.aspect_ratio_tolerance = orig_tolerance
        
        return detections
    
    def group_hvac_systems(self, detections: List[Detection],
                          system_distance: float = 100.0) -> List[List[Detection]]:
        """
        Group HVAC symbols that likely belong to the same system.
        
        Args:
            detections: List of HVAC detections
            system_distance: Maximum distance between components of same system
            
        Returns:
            List of HVAC system groups
        """
        return self.group_nearby_detections(detections, system_distance)
    
    def estimate_hvac_coverage(self, detections: List[Detection],
                             image_shape: tuple) -> Dict:
        """
        Estimate HVAC coverage statistics.
        
        Args:
            detections: List of HVAC detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with coverage statistics
        """
        if not detections:
            return {
                'count': 0,
                'coverage_ratio': 0.0,
                'avg_distance': 0.0,
                'density': 0.0
            }
        
        total_area = image_shape[0] * image_shape[1]
        hvac_area = sum(det.area for det in detections)
        
        # Calculate average distance between HVAC components
        centers = [det.center for det in detections]
        distances = []
        
        for i, (x1, y1) in enumerate(centers):
            for x2, y2 in centers[i+1:]:
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0.0
        
        return {
            'count': len(detections),
            'coverage_ratio': hvac_area / total_area,
            'avg_distance': float(avg_distance),
            'density': len(detections) / (total_area / 1000000)  # per million pixels
        }
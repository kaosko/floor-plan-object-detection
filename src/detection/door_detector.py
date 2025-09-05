"""
Door Symbol Detector with arc validation and door-specific filtering.
"""
import cv2
import numpy as np
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection


class DoorSymbolDetector(BaseSymbolDetector):
    """Detector specialized for door symbols"""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize Door detector.
        
        Args:
            threshold: Detection confidence threshold
        """
        super().__init__("door", threshold)
        
        # Door-specific parameters
        self.min_door_width = 20
        self.max_door_width = 200
        self.door_aspect_ratios = [0.3, 2.0, 3.3]  # Various door types
        
    def get_detection_params(self) -> Dict:
        """
        Get door-specific detection parameters.
        
        Returns:
            Dictionary of detection parameters
        """
        return {
            'min_width': self.min_door_width,
            'max_width': self.max_door_width,
            'scales': [0.9, 1.0, 1.1],
            'angles': [0, 90, 180, 270],  # Doors aligned with walls
            'use_line_detection': True
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """
        Detect door symbols with line-based validation.
        
        Args:
            image: Target image
            template: Template image
            
        Returns:
            List of door detections
        """
        
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
        """
        Validate detections using line/arc detection.
        
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
            
            # Check for line patterns (door frames)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=20)
            
            # If arc/circle detected or sufficient lines, likely a door
            has_arc = circles is not None
            has_lines = lines is not None and len(lines) >= 2
            has_edges = np.sum(edges) > 100
            
            if has_arc or has_lines or has_edges:
                validated.append(det)
        
        return validated
    
    def _filter_door_specific(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply door-specific filtering.
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered detections
        """
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
    
    def detect_door_type(self, image: np.ndarray, 
                        template: np.ndarray) -> List[Dict]:
        """
        Detect doors and classify their types.
        
        Args:
            image: Target image
            template: Template image
            
        Returns:
            List of detections with door type information
        """
        detections = self.detect(image, template)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        typed_detections = []
        
        for det in detections:
            # Extract region
            roi = gray[det.y1:det.y2, det.x1:det.x2]
            
            if roi.size == 0:
                continue
            
            door_type = self._classify_door_type(roi, det)
            
            typed_detections.append({
                'detection': det,
                'door_type': door_type,
                'confidence': det.confidence
            })
        
        return typed_detections
    
    def _classify_door_type(self, roi: np.ndarray, detection: Detection) -> str:
        """
        Classify the type of door based on ROI analysis.
        
        Args:
            roi: Region of interest
            detection: Detection object
            
        Returns:
            Classified door type
        """
        width = detection.width
        height = detection.height
        aspect_ratio = width / height if height > 0 else 0
        
        # Basic classification based on aspect ratio and size
        if aspect_ratio > 2.5:
            return 'sliding_door'
        elif aspect_ratio < 0.5:
            return 'narrow_door'
        elif 0.8 <= aspect_ratio <= 1.2:
            return 'swing_door'
        elif aspect_ratio > 1.5:
            return 'double_door'
        else:
            # Analyze ROI for more specific patterns
            edges = cv2.Canny(roi, 50, 150)
            
            # Check for arc patterns (swing doors)
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=min(width, height)//2
            )
            
            if circles is not None:
                return 'swing_door'
            
            # Check for straight lines (sliding doors)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=15)
            
            if lines is not None and len(lines) >= 3:
                return 'sliding_door'
            
            return 'standard_door'
    
    def detect_door_openings(self, image: np.ndarray,
                           wall_mask: np.ndarray = None) -> List[Detection]:
        """
        Detect door openings (gaps in walls) in addition to door symbols.
        
        Args:
            image: Target image
            wall_mask: Binary mask indicating wall locations
            
        Returns:
            List of door opening detections
        """
        if wall_mask is None:
            # Simple wall detection using edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect edges (walls are typically strong edges)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect wall segments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            wall_mask = cv2.dilate(edges, kernel, iterations=2)
        
        # Find gaps in walls (potential door openings)
        # Invert wall mask to find gaps
        gap_mask = cv2.bitwise_not(wall_mask)
        
        # Find contours of gaps
        contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        door_openings = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (should be door-sized)
            if (self.min_door_width <= max(w, h) <= self.max_door_width and
                min(w, h) >= 10):  # Minimum thickness
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if any(abs(aspect_ratio - expected) < 1.0 
                       for expected in self.door_aspect_ratios):
                    
                    detection = Detection(
                        x1=x, y1=y, x2=x+w, y2=y+h,
                        confidence=0.8,  # High confidence for structural gaps
                        class_name='door_opening'
                    )
                    door_openings.append(detection)
        
        return door_openings
    
    def analyze_door_accessibility(self, detections: List[Detection],
                                 image_shape: tuple) -> Dict:
        """
        Analyze door accessibility and distribution.
        
        Args:
            detections: List of door detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with accessibility analysis
        """
        if not detections:
            return {
                'count': 0,
                'avg_width': 0.0,
                'accessibility_score': 0.0,
                'distribution': 'none'
            }
        
        widths = [det.width for det in detections]
        heights = [det.height for det in detections]
        
        # Calculate accessibility score (based on door sizes)
        # Standard accessible door width is typically 32+ inches
        # Assuming 1 pixel ≈ some real unit, we use relative scoring
        avg_width = np.mean(widths)
        accessibility_score = min(1.0, avg_width / 80.0)  # Normalize to [0, 1]
        
        # Analyze distribution
        centers = [det.center for det in detections]
        img_center = (image_shape[1] / 2, image_shape[0] / 2)
        
        distances_from_center = [
            np.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
            for cx, cy in centers
        ]
        
        avg_distance = np.mean(distances_from_center)
        max_distance = np.sqrt((image_shape[1]/2)**2 + (image_shape[0]/2)**2)
        
        if avg_distance < max_distance * 0.3:
            distribution = 'centralized'
        elif avg_distance > max_distance * 0.7:
            distribution = 'peripheral' 
        else:
            distribution = 'distributed'
        
        return {
            'count': len(detections),
            'avg_width': float(np.mean(widths)),
            'avg_height': float(np.mean(heights)),
            'accessibility_score': float(accessibility_score),
            'distribution': distribution,
            'width_std': float(np.std(widths)),
            'height_std': float(np.std(heights))
        }
    
    def group_door_systems(self, detections: List[Detection],
                          proximity_threshold: float = 100.0) -> List[List[Detection]]:
        """
        Group doors that are likely part of the same entrance system.
        
        Args:
            detections: List of door detections
            proximity_threshold: Maximum distance for grouping
            
        Returns:
            List of door groups (e.g., double doors, entry systems)
        """
        return self.group_nearby_detections(detections, proximity_threshold)
"""
Window Symbol Detector with parallel line detection and grouping capabilities.
"""
import cv2
import numpy as np
from typing import List, Dict
from src.detection.base_detector import BaseSymbolDetector
from src.models.data_models import Detection


class WindowSymbolDetector(BaseSymbolDetector):
    """Detector specialized for window symbols"""
    
    def __init__(self, threshold: float = 0.68):
        """
        Initialize Window detector.
        
        Args:
            threshold: Detection confidence threshold
        """
        super().__init__("window", threshold)
        
        # Window-specific parameters
        self.parallel_line_threshold = 0.8
        self.min_window_length = 30
        self.max_window_length = 300
        
    def get_detection_params(self) -> Dict:
        """
        Get window-specific detection parameters.
        
        Returns:
            Dictionary of detection parameters
        """
        return {
            'min_length': self.min_window_length,
            'max_length': self.max_window_length,
            'scales': [0.95, 1.0, 1.05],
            'angles': [0, 90],  # Windows typically horizontal or vertical
            'detect_parallel_lines': True
        }
    
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """
        Detect window symbols (typically parallel lines).
        
        Args:
            image: Target image
            template: Template image
            
        Returns:
            List of window detections
        """
        
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
        """
        Validate that detection contains window pattern.
        
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
            
            # Check for window characteristics
            if self._has_window_characteristics(roi):
                validated.append(det)
        
        return validated
    
    def _has_window_characteristics(self, roi: np.ndarray) -> bool:
        """
        Check if ROI has characteristics typical of windows.
        
        Args:
            roi: Region of interest
            
        Returns:
            True if ROI has window characteristics
        """
        if roi.size == 0:
            return False
        
        try:
            # Detect edges
            edges = cv2.Canny(roi, 50, 150)
            
            # Detect lines (windows often have parallel lines)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=10)
            
            if lines is None:
                return False
            
            # Check for parallel lines
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                rho, theta = line[0]
                
                # Classify as horizontal or vertical
                if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                    horizontal_lines.append((rho, theta))
                elif abs(theta - np.pi/2) < np.pi/4:
                    vertical_lines.append((rho, theta))
            
            # Windows typically have at least 2 parallel lines
            has_parallel_horizontal = len(horizontal_lines) >= 2
            has_parallel_vertical = len(vertical_lines) >= 2
            
            if has_parallel_horizontal or has_parallel_vertical:
                return True
            
            # Alternative: Check for rectangular patterns
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Check if contour is approximately rectangular
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) == 4:  # Quadrilateral (likely window)
                    return True
            
        except Exception:
            # If processing fails, use simpler validation
            pass
        
        # Fallback: Use confidence threshold
        return True  # Already passed template matching
    
    def _filter_by_window_dimensions(self, 
                                    detections: List[Detection]) -> List[Detection]:
        """
        Filter based on window-specific dimensions.
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            length = max(det.x2 - det.x1, det.y2 - det.y1)
            
            if self.min_window_length <= length <= self.max_window_length:
                filtered.append(det)
        
        return filtered
    
    def detect_window_groups(self, image: np.ndarray,
                           template: np.ndarray,
                           group_threshold: float = 50) -> List[List[Detection]]:
        """
        Detect and group aligned windows.
        
        Args:
            image: Target image
            template: Template image
            group_threshold: Maximum distance for grouping
            
        Returns:
            List of window groups
        """
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
    
    def detect_by_orientation(self, image: np.ndarray,
                            template: np.ndarray,
                            orientation: str = 'both') -> List[Detection]:
        """
        Detect windows with specific orientation.
        
        Args:
            image: Target image
            template: Template image
            orientation: 'horizontal', 'vertical', or 'both'
            
        Returns:
            List of orientation-filtered detections
        """
        if orientation == 'horizontal':
            angles = [0, 180]
        elif orientation == 'vertical':
            angles = [90, 270]
        else:  # both
            angles = [0, 90, 180, 270]
        
        matches = self.matching_engine.match_multi_angle(
            image=image,
            template=template,
            angles=angles,
            threshold=self.threshold,
            max_matches=200
        )
        
        detections = self.matches_to_detections(matches)
        
        # Additional filtering based on aspect ratio
        if orientation == 'horizontal':
            detections = [det for det in detections if det.width > det.height]
        elif orientation == 'vertical':
            detections = [det for det in detections if det.height > det.width]
        
        return detections
    
    def analyze_window_layout(self, detections: List[Detection],
                            image_shape: tuple) -> Dict:
        """
        Analyze window layout and distribution patterns.
        
        Args:
            detections: List of window detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with layout analysis
        """
        if not detections:
            return {
                'count': 0,
                'orientation_ratio': 0.0,
                'alignment_score': 0.0,
                'distribution': 'none'
            }
        
        # Analyze orientations
        horizontal_windows = [det for det in detections if det.width > det.height]
        vertical_windows = [det for det in detections if det.height > det.width]
        
        orientation_ratio = len(horizontal_windows) / len(detections)
        
        # Analyze alignment
        alignment_score = self._calculate_alignment_score(detections)
        
        # Analyze distribution
        centers = [det.center for det in detections]
        
        # Check if windows are concentrated on perimeter (typical)
        img_center = (image_shape[1] / 2, image_shape[0] / 2)
        distances_from_center = [
            np.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
            for cx, cy in centers
        ]
        
        avg_distance = np.mean(distances_from_center)
        max_distance = np.sqrt((image_shape[1]/2)**2 + (image_shape[0]/2)**2)
        
        if avg_distance > max_distance * 0.6:
            distribution = 'perimeter'
        elif avg_distance < max_distance * 0.3:
            distribution = 'central'
        else:
            distribution = 'distributed'
        
        return {
            'count': len(detections),
            'horizontal_count': len(horizontal_windows),
            'vertical_count': len(vertical_windows),
            'orientation_ratio': float(orientation_ratio),
            'alignment_score': float(alignment_score),
            'distribution': distribution,
            'avg_distance_from_center': float(avg_distance)
        }
    
    def _calculate_alignment_score(self, detections: List[Detection]) -> float:
        """
        Calculate how well-aligned the windows are.
        
        Args:
            detections: List of detections
            
        Returns:
            Alignment score (0-1, higher is better aligned)
        """
        if len(detections) < 2:
            return 1.0
        
        centers = [det.center for det in detections]
        
        # Calculate alignment for both horizontal and vertical
        y_coords = [cy for cx, cy in centers]
        x_coords = [cx for cx, cy in centers]
        
        # Standard deviation of coordinates (lower is better aligned)
        y_std = np.std(y_coords)
        x_std = np.std(x_coords)
        
        # Normalize by image dimensions for scale independence
        # (This would need actual image shape, using arbitrary normalization)
        normalized_y_std = y_std / 1000.0  # Assuming typical image size
        normalized_x_std = x_std / 1000.0
        
        # Alignment score (inverse of standard deviation)
        alignment_score = 1.0 / (1.0 + min(normalized_y_std, normalized_x_std))
        
        return alignment_score
    
    def detect_window_walls(self, image: np.ndarray,
                          detections: List[Detection]) -> List[Dict]:
        """
        Identify which walls contain windows.
        
        Args:
            image: Target image
            detections: List of window detections
            
        Returns:
            List of wall information with window counts
        """
        if not detections:
            return []
        
        # Group windows by approximate wall location
        # This is a simplified approach - in practice would need wall detection
        
        groups = self.detect_window_groups(image, None, group_threshold=100)
        
        wall_info = []
        for i, group in enumerate(groups):
            if len(group) == 0:
                continue
                
            # Calculate group statistics
            centers = [det.center for det in group]
            avg_center = (
                np.mean([cx for cx, cy in centers]),
                np.mean([cy for cx, cy in centers])
            )
            
            # Determine orientation (horizontal or vertical alignment)
            if len(group) > 1:
                x_spread = np.std([cx for cx, cy in centers])
                y_spread = np.std([cy for cx, cy in centers])
                
                if x_spread > y_spread:
                    orientation = 'horizontal_wall'
                else:
                    orientation = 'vertical_wall'
            else:
                orientation = 'single_window'
            
            wall_info.append({
                'wall_id': i,
                'window_count': len(group),
                'orientation': orientation,
                'center': avg_center,
                'windows': group
            })
        
        return wall_info
    
    def estimate_natural_light(self, detections: List[Detection],
                             image_shape: tuple) -> Dict:
        """
        Estimate natural light availability based on window distribution.
        
        Args:
            detections: List of window detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with light analysis
        """
        if not detections:
            return {
                'total_window_area': 0,
                'window_wall_ratio': 0.0,
                'light_score': 0.0,
                'coverage': 'poor'
            }
        
        # Calculate total window area
        total_window_area = sum(det.area for det in detections)
        
        # Estimate building perimeter (simplified)
        perimeter_length = 2 * (image_shape[0] + image_shape[1])
        
        # Window-to-wall ratio (rough estimate)
        window_wall_ratio = total_window_area / (perimeter_length * 20)  # Assuming wall thickness
        
        # Calculate light score based on count, area, and distribution
        count_score = min(1.0, len(detections) / 10.0)  # Normalize to 10 windows
        area_score = min(1.0, total_window_area / (image_shape[0] * image_shape[1] * 0.1))
        
        light_score = (count_score + area_score + window_wall_ratio) / 3.0
        
        # Categorize coverage
        if light_score > 0.7:
            coverage = 'excellent'
        elif light_score > 0.5:
            coverage = 'good'
        elif light_score > 0.3:
            coverage = 'fair'
        else:
            coverage = 'poor'
        
        return {
            'window_count': len(detections),
            'total_window_area': int(total_window_area),
            'window_wall_ratio': float(window_wall_ratio),
            'light_score': float(light_score),
            'coverage': coverage
        }
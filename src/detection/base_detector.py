"""
Base abstract detector class for all symbol detectors.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from src.models.data_models import Detection, DetectionConfig, Match
from src.detection.matching_engine import TemplateMatchingEngine


class BaseSymbolDetector(ABC):
    """Abstract base class for all symbol detectors"""
    
    def __init__(self, class_name: str, 
                 threshold: float = 0.65,
                 config: Optional[DetectionConfig] = None):
        """
        Initialize the base detector.
        
        Args:
            class_name: Name of the symbol class
            threshold: Detection confidence threshold
            config: Detection configuration
        """
        self.class_name = class_name
        self.threshold = threshold
        self.config = config or DetectionConfig(pdf_path='')
        self.matching_engine = TemplateMatchingEngine(self.config.method)
        self.detections = []
        
    @abstractmethod
    def detect(self, image: np.ndarray, 
              template: np.ndarray) -> List[Detection]:
        """
        Detect symbols in image using template.
        
        Args:
            image: Target image
            template: Template image
            
        Returns:
            List of detections
        """
        pass
    
    @abstractmethod
    def get_detection_params(self) -> Dict:
        """
        Get detector-specific parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    def matches_to_detections(self, matches: List[Match]) -> List[Detection]:
        """
        Convert Match objects to Detection objects.
        
        Args:
            matches: List of template matches
            
        Returns:
            List of detections
        """
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
        """
        Apply Non-Maximum Suppression to remove duplicates.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered detections
        """
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
        """
        Export detections in YOLO format.
        
        Args:
            detections: List of detections
            image_shape: Shape of the image (height, width)
            output_path: Output file path
            class_id: Class ID for YOLO format
        """
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
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detections
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            Filtered detections
        """
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
        """
        Filter detections by aspect ratio.
        
        Args:
            detections: List of detections
            min_ratio: Minimum aspect ratio
            max_ratio: Maximum aspect ratio
            
        Returns:
            Filtered detections
        """
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
    
    def filter_by_confidence(self, detections: List[Detection],
                           min_confidence: float) -> List[Detection]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered detections
        """
        return [det for det in detections if det.confidence >= min_confidence]
    
    def group_nearby_detections(self, detections: List[Detection],
                              distance_threshold: float = 50.0) -> List[List[Detection]]:
        """
        Group detections that are close to each other.
        
        Args:
            detections: List of detections
            distance_threshold: Maximum distance between centers
            
        Returns:
            List of detection groups
        """
        if not detections:
            return []
        
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
                    
                # Calculate distance between centers
                cx1, cy1 = det1.center
                cx2, cy2 = det2.center
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                if distance <= distance_threshold:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def validate_detections(self, detections: List[Detection],
                          image: np.ndarray) -> List[Detection]:
        """
        Validate detections using image context.
        Base implementation - can be overridden by subclasses.
        
        Args:
            detections: List of detections
            image: Source image
            
        Returns:
            Validated detections
        """
        # Base validation - just check bounds
        validated = []
        h, w = image.shape[:2]
        
        for det in detections:
            if (0 <= det.x1 < det.x2 <= w and 
                0 <= det.y1 < det.y2 <= h):
                validated.append(det)
        
        return validated
    
    def get_detection_statistics(self, detections: List[Detection]) -> Dict:
        """
        Get statistics about the detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {
                'count': 0,
                'mean_confidence': 0.0,
                'std_confidence': 0.0,
                'mean_area': 0.0,
                'std_area': 0.0
            }
        
        confidences = [det.confidence for det in detections]
        areas = [det.area for det in detections]
        
        return {
            'count': len(detections),
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'min_area': int(np.min(areas)),
            'max_area': int(np.max(areas))
        }
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[Detection],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw detections on image for visualization.
        
        Args:
            image: Input image
            detections: List of detections
            color: Bounding box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with detections drawn
        """
        result = image.copy()
        
        for det in detections:
            # Draw bounding box
            cv2.rectangle(result, (det.x1, det.y1), (det.x2, det.y2), 
                         color, thickness)
            
            # Draw confidence score
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(result, (det.x1, det.y1 - label_size[1] - 5),
                         (det.x1 + label_size[0], det.y1), color, -1)
            
            # Text
            cv2.putText(result, label, (det.x1, det.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
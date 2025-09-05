#!/usr/bin/env python3
"""
Data classes for Analytical Symbol Detection System
Contains data structures for detections, results, and configurations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time


@dataclass
class Detection:
    """Single detection result"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    scale: float = 1.0
    angle: float = 0.0
    
    @property
    def center_x(self) -> float:
        """Get center X coordinate"""
        return (self.x1 + self.x2) / 2.0
    
    @property
    def center_y(self) -> float:
        """Get center Y coordinate"""
        return (self.y1 + self.y2) / 2.0
    
    @property
    def width(self) -> int:
        """Get detection width"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get detection height"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get detection area"""
        return self.width * self.height
    
    def to_yolo_format(self, image_width: int, image_height: int, class_id: int = 0) -> str:
        """
        Convert detection to YOLO format string
        
        Args:
            image_width: Full image width
            image_height: Full image height
            class_id: Class ID for YOLO format
            
        Returns:
            YOLO format string
        """
        center_x_norm = self.center_x / image_width
        center_y_norm = self.center_y / image_height
        width_norm = self.width / image_width
        height_norm = self.height / image_height
        
        return f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'confidence': self.confidence,
            'class_name': self.class_name,
            'scale': self.scale,
            'angle': self.angle
        }


@dataclass
class DetectionResults:
    """Complete detection results"""
    detections: List[Detection]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.metadata:
            self.metadata = {}
        
        # Add summary statistics
        self.metadata.update({
            'num_detections': len(self.detections),
            'timestamp': time.time(),
            'avg_confidence': self.get_average_confidence(),
            'confidence_range': self.get_confidence_range(),
            'detection_areas': [det.area for det in self.detections],
            'scales_used': list(set(det.scale for det in self.detections)),
            'angles_used': list(set(det.angle for det in self.detections))
        })
    
    def get_average_confidence(self) -> float:
        """Get average confidence of all detections"""
        if not self.detections:
            return 0.0
        return sum(det.confidence for det in self.detections) / len(self.detections)
    
    def get_confidence_range(self) -> Tuple[float, float]:
        """Get min and max confidence values"""
        if not self.detections:
            return (0.0, 0.0)
        
        confidences = [det.confidence for det in self.detections]
        return (min(confidences), max(confidences))
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResults':
        """
        Filter detections by minimum confidence
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            New DetectionResults with filtered detections
        """
        filtered_detections = [det for det in self.detections if det.confidence >= min_confidence]
        
        new_results = DetectionResults(
            detections=filtered_detections,
            processing_time=self.processing_time,
            metadata=self.metadata.copy()
        )
        new_results.metadata['filter_applied'] = f"confidence >= {min_confidence}"
        
        return new_results
    
    def sort_by_confidence(self, ascending: bool = False) -> 'DetectionResults':
        """
        Sort detections by confidence
        
        Args:
            ascending: If True, sort in ascending order
            
        Returns:
            New DetectionResults with sorted detections
        """
        sorted_detections = sorted(self.detections, 
                                 key=lambda det: det.confidence, 
                                 reverse=not ascending)
        
        return DetectionResults(
            detections=sorted_detections,
            processing_time=self.processing_time,
            metadata=self.metadata.copy()
        )
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class"""
        return [det for det in self.detections if det.class_name == class_name]
    
    def to_yolo_format(self, image_width: int, image_height: int, 
                      class_mapping: Optional[Dict[str, int]] = None) -> List[str]:
        """
        Convert all detections to YOLO format
        
        Args:
            image_width: Full image width
            image_height: Full image height
            class_mapping: Optional mapping from class names to IDs
            
        Returns:
            List of YOLO format strings
        """
        yolo_lines = []
        
        for det in self.detections:
            if class_mapping:
                class_id = class_mapping.get(det.class_name, 0)
            else:
                class_id = 0
            
            yolo_line = det.to_yolo_format(image_width, image_height, class_id)
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    
    def export_summary(self) -> Dict[str, Any]:
        """Export summary statistics"""
        return {
            'total_detections': len(self.detections),
            'processing_time': self.processing_time,
            'average_confidence': self.get_average_confidence(),
            'confidence_range': self.get_confidence_range(),
            'class_distribution': self.get_class_distribution(),
            'metadata': self.metadata
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of detections by class"""
        class_counts = {}
        for det in self.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        return class_counts


@dataclass 
class TemplateMatchCandidate:
    """Candidate from template matching"""
    x: int
    y: int
    width: int
    height: int
    score: float
    scale: float
    angle: float
    
    @property
    def x1(self) -> int:
        return self.x
    
    @property
    def y1(self) -> int:
        return self.y
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    def to_detection(self, class_name: str) -> Detection:
        """Convert to Detection object"""
        return Detection(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            confidence=self.score,
            class_name=class_name,
            scale=self.scale,
            angle=self.angle
        )


@dataclass
class ProcessingStats:
    """Processing statistics for pipeline stages"""
    stage_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self) -> None:
        """Mark stage as finished and calculate duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stage_name': self.stage_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_usage': self.memory_usage,
            'additional_info': self.additional_info
        }


@dataclass
class PipelineResults:
    """Complete pipeline execution results"""
    detection_results: DetectionResults
    processing_stats: List[ProcessingStats]
    template_metadata: Dict[str, Any]
    config_used: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    
    def get_total_processing_time(self) -> float:
        """Get total processing time across all stages"""
        return sum(stat.duration for stat in self.processing_stats if stat.duration)
    
    def get_stage_durations(self) -> Dict[str, float]:
        """Get durations for each processing stage"""
        return {stat.stage_name: stat.duration for stat in self.processing_stats if stat.duration}
    
    def export_report(self) -> Dict[str, Any]:
        """Export comprehensive results report"""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'detection_summary': self.detection_results.export_summary() if self.detection_results else {},
            'total_processing_time': self.get_total_processing_time(),
            'stage_durations': self.get_stage_durations(),
            'template_metadata': self.template_metadata,
            'config_used': self.config_used,
            'processing_stats': [stat.to_dict() for stat in self.processing_stats]
        }
#!/usr/bin/env python3
"""
Integrated Floorplan Analyzer

Combines room detection, door/window detection, and connection analysis
into a unified workflow for complete floor plan understanding.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json


@dataclass
class FloorplanAnalysisResult:
    """Complete results from floorplan analysis"""
    # Image info
    image_shape: Tuple[int, ...]
    
    # Detection results
    floorplan_contour: Optional[Any] = None
    rooms: List[Any] = field(default_factory=list)
    door_detections: List[Any] = field(default_factory=list)
    window_detections: List[Any] = field(default_factory=list)
    
    # Connection analysis
    opening_connections: List[Any] = field(default_factory=list)
    room_adjacency_matrix: Optional[np.ndarray] = None
    
    # Processing info
    processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'image_shape': list(self.image_shape),
            'processing_time': self.processing_time,
            'stage_times': self.stage_times,
            'summary': {
                'num_rooms': len(self.rooms),
                'num_doors': len(self.door_detections),
                'num_windows': len(self.window_detections),
                'num_connections': len(self.opening_connections)
            }
        }
        
        # Floorplan
        if self.floorplan_contour:
            result['floorplan'] = self.floorplan_contour.to_dict()
        
        # Rooms
        result['rooms'] = [room.to_dict() for room in self.rooms]
        
        # Doors
        result['doors'] = [
            {
                'index': i,
                'bbox': [d.x1, d.y1, d.x2, d.y2],
                'center': [d.center_x, d.center_y],
                'confidence': d.confidence,
                'class': d.class_name
            }
            for i, d in enumerate(self.door_detections)
        ]
        
        # Windows
        result['windows'] = [
            {
                'index': i,
                'bbox': [w.x1, w.y1, w.x2, w.y2],
                'center': [w.center_x, w.center_y],
                'confidence': w.confidence,
                'class': w.class_name
            }
            for i, w in enumerate(self.window_detections)
        ]
        
        # Connections
        result['connections'] = [
            conn.to_dict() for conn in self.opening_connections
        ]
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get complete info for a specific room"""
        for room in self.rooms:
            if room.id == room_id:
                room_doors = [
                    conn for conn in self.opening_connections
                    if room_id in conn.room_ids and 
                    conn.opening_type.value == 'door'
                ]
                room_windows = [
                    conn for conn in self.opening_connections
                    if room_id in conn.room_ids and 
                    conn.opening_type.value == 'window'
                ]
                
                return {
                    'room': room.to_dict(),
                    'doors': [d.to_dict() for d in room_doors],
                    'windows': [w.to_dict() for w in room_windows],
                    'num_doors': len(room_doors),
                    'num_windows': len(room_windows)
                }
        return None


class FloorplanAnalyzer:
    """
    Main class for complete floorplan analysis.
    
    Integrates:
    - Room/boundary detection
    - Door detection
    - Window detection
    - Opening-to-room connection analysis
    """
    
    def __init__(self,
                 door_detector: Optional[Any] = None,
                 window_detector: Optional[Any] = None,
                 min_room_area: int = 5000,
                 min_floorplan_area: int = 50000,
                 proximity_threshold: int = 50):
        """
        Initialize the analyzer.
        
        Args:
            door_detector: Optional pre-configured door detector
            window_detector: Optional pre-configured window detector
            min_room_area: Minimum room area in pixels
            min_floorplan_area: Minimum floorplan area in pixels
            proximity_threshold: Max distance for opening-room association
        """
        from src.detection.room_detector import RoomDetector
        from src.detection.opening_connection_analyzer import OpeningConnectionAnalyzer
        
        self.room_detector = RoomDetector(
            min_room_area=min_room_area,
            min_floorplan_area=min_floorplan_area
        )
        self.connection_analyzer = OpeningConnectionAnalyzer(
            proximity_threshold=proximity_threshold
        )
        
        self.door_detector = door_detector
        self.window_detector = window_detector
        
        # Results storage
        self.last_result: Optional[FloorplanAnalysisResult] = None
    
    def analyze(self, 
                image: np.ndarray,
                door_template: Optional[np.ndarray] = None,
                window_template: Optional[np.ndarray] = None,
                existing_door_detections: Optional[List] = None,
                existing_window_detections: Optional[List] = None) -> FloorplanAnalysisResult:
        """
        Perform complete floorplan analysis.
        
        Args:
            image: Input floor plan image
            door_template: Optional template for door detection
            window_template: Optional template for window detection
            existing_door_detections: Pre-computed door detections (skips detection)
            existing_window_detections: Pre-computed window detections (skips detection)
            
        Returns:
            FloorplanAnalysisResult with complete analysis
        """
        start_time = time.time()
        stage_times = {}
        
        # Stage 1: Room and boundary detection
        stage_start = time.time()
        floorplan_contour, rooms = self.room_detector.detect(image)
        stage_times['room_detection'] = time.time() - stage_start
        
        # Stage 2: Door detection
        stage_start = time.time()
        if existing_door_detections is not None:
            door_detections = existing_door_detections
        elif self.door_detector and door_template is not None:
            door_detections = self.door_detector.detect(image, door_template)
        else:
            door_detections = []
        stage_times['door_detection'] = time.time() - stage_start
        
        # Stage 3: Window detection
        stage_start = time.time()
        if existing_window_detections is not None:
            window_detections = existing_window_detections
        elif self.window_detector and window_template is not None:
            window_detections = self.window_detector.detect(image, window_template)
        else:
            window_detections = []
        stage_times['window_detection'] = time.time() - stage_start
        
        # Stage 4: Connection analysis
        stage_start = time.time()
        connections = self.connection_analyzer.analyze(
            door_detections=door_detections,
            window_detections=window_detections,
            rooms=rooms,
            floorplan_contour=floorplan_contour
        )
        stage_times['connection_analysis'] = time.time() - stage_start
        
        # Stage 5: Generate adjacency matrix
        stage_start = time.time()
        adjacency_matrix = None
        if rooms:
            adjacency_matrix = self.connection_analyzer.generate_adjacency_matrix(rooms)
        stage_times['adjacency_calculation'] = time.time() - stage_start
        
        total_time = time.time() - start_time
        
        # Build result
        result = FloorplanAnalysisResult(
            image_shape=image.shape,
            floorplan_contour=floorplan_contour,
            rooms=rooms,
            door_detections=door_detections,
            window_detections=window_detections,
            opening_connections=connections,
            room_adjacency_matrix=adjacency_matrix,
            processing_time=total_time,
            stage_times=stage_times
        )
        
        self.last_result = result
        return result
    
    def visualize(self, 
                  image: np.ndarray,
                  result: Optional[FloorplanAnalysisResult] = None,
                  show_floorplan: bool = True,
                  show_rooms: bool = True,
                  show_doors: bool = True,
                  show_windows: bool = True,
                  show_connections: bool = True,
                  show_labels: bool = True) -> np.ndarray:
        """
        Visualize analysis results on the image.
        
        Args:
            image: Original image
            result: Analysis result (uses last_result if None)
            show_floorplan: Draw floorplan boundary
            show_rooms: Draw room boundaries
            show_doors: Draw door detections
            show_windows: Draw window detections
            show_connections: Draw connection lines
            show_labels: Show text labels
            
        Returns:
            Annotated image
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            return image.copy()
        
        # Start with room visualization
        vis = self.room_detector.visualize(
            image, 
            show_floorplan=show_floorplan,
            show_rooms=show_rooms,
            show_labels=show_labels
        )
        
        # Draw doors
        if show_doors:
            for door in result.door_detections:
                cv2.rectangle(vis, (door.x1, door.y1), (door.x2, door.y2), 
                            (0, 255, 0), 2)
                if show_labels:
                    label = f"Door ({door.confidence:.2f})"
                    cv2.putText(vis, label, (door.x1, door.y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw windows  
        if show_windows:
            for window in result.window_detections:
                cv2.rectangle(vis, (window.x1, window.y1), (window.x2, window.y2),
                            (255, 165, 0), 2)
                if show_labels:
                    label = f"Window ({window.confidence:.2f})"
                    cv2.putText(vis, label, (window.x1, window.y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # Draw connections
        if show_connections:
            for conn in result.opening_connections:
                cx, cy = int(conn.center[0]), int(conn.center[1])
                
                # Color based on connection type
                if conn.is_exterior:
                    color = (0, 0, 255)  # Red for exterior
                elif len(conn.room_ids) == 2:
                    color = (255, 255, 0)  # Cyan for room-to-room
                else:
                    color = (128, 128, 128)  # Gray for other
                
                # Draw lines to connected rooms
                for room_id in conn.room_ids:
                    for room in result.rooms:
                        if room.id == room_id:
                            rx, ry = int(room.centroid[0]), int(room.centroid[1])
                            cv2.line(vis, (cx, cy), (rx, ry), color, 1)
                
                # Draw connection label
                if show_labels and conn.room_ids:
                    desc = conn.get_description()
                    # Truncate long descriptions
                    if len(desc) > 30:
                        desc = desc[:27] + "..."
                    cv2.putText(vis, desc, (cx - 40, cy + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return vis
    
    def export_to_geojson(self, 
                          result: Optional[FloorplanAnalysisResult] = None,
                          pixel_to_meter: float = 0.01) -> Dict[str, Any]:
        """
        Export results in GeoJSON-like format.
        
        Args:
            result: Analysis result (uses last_result if None)
            pixel_to_meter: Scale factor for pixel to meter conversion
            
        Returns:
            GeoJSON-like dictionary
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            return {}
        
        features = []
        
        # Add rooms as polygons
        for room in result.rooms:
            contour_points = room.contour.points.squeeze().tolist()
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [contour_points]
                },
                'properties': {
                    'id': room.id,
                    'type': 'room',
                    'room_type': room.room_type.value,
                    'area_px': room.area,
                    'area_m2': room.area * (pixel_to_meter ** 2),
                    'connected_doors': room.connected_doors,
                    'adjacent_rooms': room.adjacent_rooms
                }
            }
            features.append(feature)
        
        # Add openings as points
        for conn in result.opening_connections:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': list(conn.center)
                },
                'properties': {
                    'id': f"opening_{conn.opening_index}",
                    'type': conn.opening_type.value,
                    'connection_type': conn.connection_type.value,
                    'connected_rooms': conn.room_ids,
                    'is_exterior': conn.is_exterior,
                    'confidence': conn.confidence
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'pixel_to_meter': pixel_to_meter,
                'processing_time': result.processing_time
            }
        }
    
    def print_summary(self, result: Optional[FloorplanAnalysisResult] = None) -> None:
        """Print a human-readable summary of the analysis"""
        if result is None:
            result = self.last_result
        
        if result is None:
            print("No analysis results available.")
            return
        
        print("\n" + "=" * 60)
        print("FLOORPLAN ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nImage Shape: {result.image_shape}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        print(f"\n--- Detections ---")
        print(f"Rooms detected: {len(result.rooms)}")
        print(f"Doors detected: {len(result.door_detections)}")
        print(f"Windows detected: {len(result.window_detections)}")
        
        print(f"\n--- Rooms ---")
        for room in result.rooms:
            print(f"  {room.id}: {room.room_type.value}")
            print(f"    Area: {room.area:.0f} px²")
            print(f"    Doors: {len(room.connected_doors)}")
            print(f"    Adjacent to: {room.adjacent_rooms}")
        
        print(f"\n--- Door Connections ---")
        for conn in result.opening_connections:
            if conn.opening_type.value == 'door':
                print(f"  Door {conn.opening_index}: {conn.get_description()}")
        
        exterior = [c for c in result.opening_connections if c.is_exterior]
        if exterior:
            print(f"\n--- Exterior Openings ---")
            for conn in exterior:
                print(f"  {conn.opening_type.value.title()} {conn.opening_index}")
        
        print("\n" + "=" * 60)

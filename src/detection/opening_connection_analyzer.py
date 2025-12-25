#!/usr/bin/env python3
"""
Opening Connection Analyzer

Analyzes and establishes connections between detected openings (doors/windows)
and detected rooms/units in floor plans.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OpeningType(Enum):
    """Type of architectural opening"""
    DOOR = "door"
    WINDOW = "window"
    SLIDING_DOOR = "sliding_door"
    DOUBLE_DOOR = "double_door"
    POCKET_DOOR = "pocket_door"
    BIFOLD_DOOR = "bifold_door"


class ConnectionType(Enum):
    """Type of spatial connection an opening provides"""
    EXTERNAL = "external"  # Leads outside the floorplan
    ROOM_ENTRY = "room_entry"  # Single room access point
    ROOM_TO_ROOM = "room_to_room"  # Connects exactly two rooms
    JUNCTION = "junction"  # Connects multiple spaces
    UNKNOWN = "unknown"


@dataclass
class OpeningConnection:
    """Represents a connection between an opening and rooms"""
    opening_index: int
    opening_type: OpeningType
    center: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    
    # Connected spaces
    room_ids: List[str] = field(default_factory=list)
    connection_type: ConnectionType = ConnectionType.UNKNOWN
    
    # Additional metadata
    is_exterior: bool = False
    wall_direction: Optional[str] = None  # 'horizontal', 'vertical', or 'diagonal'
    
    def __post_init__(self):
        """Determine connection type based on connected rooms"""
        if not self.room_ids:
            self.connection_type = ConnectionType.EXTERNAL
            self.is_exterior = True
        elif len(self.room_ids) == 1:
            self.connection_type = ConnectionType.ROOM_ENTRY
        elif len(self.room_ids) == 2:
            self.connection_type = ConnectionType.ROOM_TO_ROOM
        else:
            self.connection_type = ConnectionType.JUNCTION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'opening_index': self.opening_index,
            'opening_type': self.opening_type.value,
            'center': self.center,
            'bounding_box': self.bounding_box,
            'confidence': self.confidence,
            'room_ids': self.room_ids,
            'connection_type': self.connection_type.value,
            'is_exterior': self.is_exterior,
            'wall_direction': self.wall_direction,
            'description': self.get_description()
        }
    
    def get_description(self) -> str:
        """Get human-readable description of the connection"""
        opening_name = self.opening_type.value.replace('_', ' ').title()
        
        if self.connection_type == ConnectionType.EXTERNAL:
            return f"{opening_name} - External access"
        elif self.connection_type == ConnectionType.ROOM_ENTRY:
            return f"{opening_name} - Entry to {self.room_ids[0]}"
        elif self.connection_type == ConnectionType.ROOM_TO_ROOM:
            return f"{opening_name} - Connects {self.room_ids[0]} ↔ {self.room_ids[1]}"
        elif self.connection_type == ConnectionType.JUNCTION:
            rooms_str = ", ".join(self.room_ids)
            return f"{opening_name} - Junction connecting: {rooms_str}"
        else:
            return f"{opening_name} - Unknown connection"


class OpeningConnectionAnalyzer:
    """
    Analyzes spatial relationships between detected openings and rooms.
    
    This class takes detected doors/windows and rooms, then determines
    which rooms each opening provides access to.
    """
    
    def __init__(self, 
                 proximity_threshold: int = 50,
                 overlap_threshold: float = 0.1):
        """
        Initialize the analyzer.
        
        Args:
            proximity_threshold: Max distance (pixels) for opening-room association
            overlap_threshold: Min overlap ratio for direct containment
        """
        self.proximity_threshold = proximity_threshold
        self.overlap_threshold = overlap_threshold
        self.connections: List[OpeningConnection] = []
    
    def analyze(self, 
                door_detections: List[Any],
                window_detections: List[Any],
                rooms: List[Any],
                floorplan_contour: Optional[Any] = None) -> List[OpeningConnection]:
        """
        Analyze all openings and establish room connections.
        
        Args:
            door_detections: List of door Detection objects
            window_detections: List of window Detection objects  
            rooms: List of Room objects from RoomDetector
            floorplan_contour: Optional FloorplanContour for exterior detection
            
        Returns:
            List of OpeningConnection objects
        """
        self.connections = []
        
        # Process doors
        for idx, door in enumerate(door_detections):
            connection = self._analyze_opening(
                idx, door, OpeningType.DOOR, rooms, floorplan_contour
            )
            self.connections.append(connection)
        
        # Process windows (with offset index)
        offset = len(door_detections)
        for idx, window in enumerate(window_detections):
            connection = self._analyze_opening(
                offset + idx, window, OpeningType.WINDOW, rooms, floorplan_contour
            )
            self.connections.append(connection)
        
        return self.connections
    
    def _analyze_opening(self,
                        index: int,
                        opening: Any,
                        opening_type: OpeningType,
                        rooms: List[Any],
                        floorplan_contour: Optional[Any]) -> OpeningConnection:
        """
        Analyze a single opening's room connections.
        
        Args:
            index: Opening index
            opening: Detection object
            opening_type: Type of opening
            rooms: List of rooms
            floorplan_contour: Floorplan boundary
            
        Returns:
            OpeningConnection with room associations
        """
        center = (opening.center_x, opening.center_y)
        bbox = (opening.x1, opening.y1, opening.x2, opening.y2)
        
        connected_rooms = []
        
        for room in rooms:
            if self._is_opening_connected_to_room(opening, room):
                connected_rooms.append(room.id)
        
        # Determine if exterior
        is_exterior = False
        if floorplan_contour and not connected_rooms:
            is_exterior = self._is_on_exterior(opening, floorplan_contour)
        
        # Detect wall direction based on opening dimensions
        wall_direction = self._detect_wall_direction(opening)
        
        connection = OpeningConnection(
            opening_index=index,
            opening_type=opening_type,
            center=center,
            bounding_box=bbox,
            confidence=opening.confidence,
            room_ids=connected_rooms,
            is_exterior=is_exterior,
            wall_direction=wall_direction
        )
        
        return connection
    
    def _is_opening_connected_to_room(self, opening: Any, room: Any) -> bool:
        """
        Determine if an opening is connected to a room.
        
        Uses multiple strategies:
        1. Direct overlap/containment
        2. Proximity to room boundary
        3. Extended bounding box overlap
        
        Args:
            opening: Detection object
            room: Room object
            
        Returns:
            True if connected
        """
        # Strategy 1: Check if opening center is inside room
        center = (opening.center_x, opening.center_y)
        if room.contains_point(center):
            return True
        
        # Strategy 2: Check if opening bbox overlaps with room bbox
        opening_bbox = (opening.x1, opening.y1, 
                       opening.x2 - opening.x1, opening.y2 - opening.y1)
        room_bbox = room.bounding_box
        
        if self._boxes_overlap(opening_bbox, room_bbox):
            return True
        
        # Strategy 3: Check proximity - is opening edge near room boundary?
        if self._is_near_room_boundary(opening, room, self.proximity_threshold):
            return True
        
        return False
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple) -> bool:
        """Check if two bounding boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or
                   y1 + h1 < y2 or y2 + h2 < y1)
    
    def _is_near_room_boundary(self, opening: Any, room: Any, 
                               threshold: int) -> bool:
        """
        Check if opening is near room boundary.
        
        Args:
            opening: Detection object
            room: Room object
            threshold: Distance threshold in pixels
            
        Returns:
            True if near boundary
        """
        try:
            import cv2
            
            # Check corners and edge midpoints of the opening
            test_points = [
                (opening.x1, opening.y1),  # Top-left
                (opening.x2, opening.y1),  # Top-right
                (opening.x1, opening.y2),  # Bottom-left
                (opening.x2, opening.y2),  # Bottom-right
                (opening.center_x, opening.y1),  # Top-center
                (opening.center_x, opening.y2),  # Bottom-center
                (opening.x1, opening.center_y),  # Left-center
                (opening.x2, opening.center_y),  # Right-center
            ]
            
            for point in test_points:
                dist = cv2.pointPolygonTest(room.contour.points, point, True)
                if abs(dist) <= threshold:
                    return True
            
            return False
        except Exception:
            # Fallback if OpenCV operation fails
            return False
    
    def _is_on_exterior(self, opening: Any, floorplan_contour: Any) -> bool:
        """
        Check if opening is on the exterior boundary.
        
        Args:
            opening: Detection object
            floorplan_contour: FloorplanContour object
            
        Returns:
            True if on exterior
        """
        try:
            import cv2
            
            center = (opening.center_x, opening.center_y)
            dist = cv2.pointPolygonTest(
                floorplan_contour.contour.points, center, True
            )
            
            # If center is inside but close to boundary, it's exterior
            if 0 < dist < self.proximity_threshold:
                return True
            
            return False
        except Exception:
            return False
    
    def _detect_wall_direction(self, opening: Any) -> str:
        """
        Detect wall direction based on opening dimensions.
        
        Args:
            opening: Detection object
            
        Returns:
            'horizontal', 'vertical', or 'diagonal'
        """
        width = opening.x2 - opening.x1
        height = opening.y2 - opening.y1
        
        aspect = width / max(height, 1)
        
        if aspect > 1.5:
            return 'horizontal'
        elif aspect < 0.67:
            return 'vertical'
        else:
            return 'diagonal'
    
    def get_room_doors(self, room_id: str) -> List[OpeningConnection]:
        """
        Get all doors connected to a specific room.
        
        Args:
            room_id: Room identifier
            
        Returns:
            List of door connections for this room
        """
        return [conn for conn in self.connections 
                if room_id in conn.room_ids and 
                conn.opening_type == OpeningType.DOOR]
    
    def get_room_windows(self, room_id: str) -> List[OpeningConnection]:
        """
        Get all windows connected to a specific room.
        
        Args:
            room_id: Room identifier
            
        Returns:
            List of window connections for this room
        """
        return [conn for conn in self.connections 
                if room_id in conn.room_ids and 
                conn.opening_type == OpeningType.WINDOW]
    
    def get_exterior_openings(self) -> List[OpeningConnection]:
        """Get all exterior openings"""
        return [conn for conn in self.connections if conn.is_exterior]
    
    def get_room_to_room_connections(self) -> List[OpeningConnection]:
        """Get all openings that connect exactly two rooms"""
        return [conn for conn in self.connections 
                if conn.connection_type == ConnectionType.ROOM_TO_ROOM]
    
    def generate_adjacency_matrix(self, rooms: List[Any]) -> np.ndarray:
        """
        Generate room adjacency matrix based on door connections.
        
        Args:
            rooms: List of Room objects
            
        Returns:
            NxN adjacency matrix where N is number of rooms
        """
        n = len(rooms)
        room_indices = {room.id: i for i, room in enumerate(rooms)}
        
        matrix = np.zeros((n, n), dtype=int)
        
        for conn in self.connections:
            if conn.connection_type == ConnectionType.ROOM_TO_ROOM:
                i = room_indices.get(conn.room_ids[0])
                j = room_indices.get(conn.room_ids[1])
                if i is not None and j is not None:
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        
        return matrix
    
    def export_results(self) -> Dict[str, Any]:
        """
        Export all connection analysis results.
        
        Returns:
            Dictionary with complete analysis results
        """
        stats = {
            'total_openings': len(self.connections),
            'doors': len([c for c in self.connections 
                         if c.opening_type == OpeningType.DOOR]),
            'windows': len([c for c in self.connections 
                           if c.opening_type == OpeningType.WINDOW]),
            'exterior_openings': len(self.get_exterior_openings()),
            'room_to_room': len(self.get_room_to_room_connections()),
        }
        
        return {
            'statistics': stats,
            'connections': [conn.to_dict() for conn in self.connections]
        }

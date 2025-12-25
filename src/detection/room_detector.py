#!/usr/bin/env python3
"""
Room and Floorplan Contour Detector

Detects:
1. Floorplan outer boundary (level contour)
2. Individual rooms/units within the floorplan
3. Provides spatial relationships between rooms and openings (doors/windows)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RoomType(Enum):
    """Classification of room types based on characteristics"""
    UNKNOWN = "unknown"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    HALLWAY = "hallway"
    CLOSET = "closet"
    BALCONY = "balcony"
    UTILITY = "utility"


@dataclass
class Contour:
    """Represents a detected contour (boundary)"""
    points: np.ndarray  # Contour points (N, 1, 2) format from OpenCV
    area: float
    perimeter: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    centroid: Tuple[float, float]
    is_closed: bool = True
    hierarchy_level: int = 0  # 0 = outermost, higher = nested
    parent_index: Optional[int] = None
    children_indices: List[int] = field(default_factory=list)
    
    @property
    def x(self) -> int:
        return self.bounding_box[0]
    
    @property
    def y(self) -> int:
        return self.bounding_box[1]
    
    @property
    def width(self) -> int:
        return self.bounding_box[2]
    
    @property
    def height(self) -> int:
        return self.bounding_box[3]
    
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
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside this contour"""
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'area': self.area,
            'perimeter': self.perimeter,
            'bounding_box': self.bounding_box,
            'centroid': self.centroid,
            'is_closed': self.is_closed,
            'hierarchy_level': self.hierarchy_level,
            'parent_index': self.parent_index,
            'children_indices': self.children_indices
        }


@dataclass
class Room:
    """Represents a detected room/unit"""
    id: str
    contour: Contour
    room_type: RoomType = RoomType.UNKNOWN
    label: Optional[str] = None  # Text label if detected
    connected_doors: List[int] = field(default_factory=list)  # Door detection indices
    connected_windows: List[int] = field(default_factory=list)  # Window detection indices
    adjacent_rooms: List[str] = field(default_factory=list)  # IDs of adjacent rooms
    
    @property
    def area(self) -> float:
        return self.contour.area
    
    @property
    def centroid(self) -> Tuple[float, float]:
        return self.contour.centroid
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        return self.contour.bounding_box
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside this room"""
        return self.contour.contains_point(point)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'room_type': self.room_type.value,
            'label': self.label,
            'area': self.area,
            'centroid': self.centroid,
            'bounding_box': self.bounding_box,
            'connected_doors': self.connected_doors,
            'connected_windows': self.connected_windows,
            'adjacent_rooms': self.adjacent_rooms,
            'contour': self.contour.to_dict()
        }


@dataclass
class FloorplanContour:
    """Represents the outer boundary of the entire floorplan/level"""
    contour: Contour
    total_area: float
    rooms: List[Room] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_area': self.total_area,
            'contour': self.contour.to_dict(),
            'num_rooms': len(self.rooms),
            'rooms': [room.to_dict() for room in self.rooms]
        }


class RoomDetector:
    """
    Detects rooms and floorplan boundaries from floor plan images.
    
    Uses contour detection and hierarchical analysis to identify:
    - Outer floorplan boundary
    - Individual room boundaries
    - Room relationships and adjacencies
    """
    
    def __init__(self, 
                 min_room_area: int = 5000,
                 min_floorplan_area: int = 50000,
                 wall_thickness_range: Tuple[int, int] = (3, 30),
                 use_adaptive_threshold: bool = True):
        """
        Initialize Room Detector.
        
        Args:
            min_room_area: Minimum area in pixels for a valid room
            min_floorplan_area: Minimum area for outer floorplan boundary
            wall_thickness_range: Expected wall thickness range (min, max) in pixels
            use_adaptive_threshold: Use adaptive thresholding for better wall detection
        """
        self.min_room_area = min_room_area
        self.min_floorplan_area = min_floorplan_area
        self.wall_thickness_range = wall_thickness_range
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # Detection results
        self.floorplan_contour: Optional[FloorplanContour] = None
        self.rooms: List[Room] = []
        self.all_contours: List[Contour] = []
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[FloorplanContour], List[Room]]:
        """
        Detect floorplan contour and rooms from image.
        
        Args:
            image: Input floor plan image (BGR or grayscale)
            
        Returns:
            Tuple of (FloorplanContour, List[Room])
        """
        # Preprocess image
        binary = self._preprocess_image(image)
        
        # Find all contours with hierarchy
        contours, hierarchy = self._find_contours(binary)
        
        if len(contours) == 0:
            return None, []
        
        # Process contours into Contour objects
        self.all_contours = self._process_contours(contours, hierarchy)
        
        # Identify floorplan boundary (largest outer contour)
        self.floorplan_contour = self._find_floorplan_boundary(self.all_contours)
        
        # Identify individual rooms
        self.rooms = self._find_rooms(self.all_contours, binary)
        
        # Assign rooms to floorplan
        if self.floorplan_contour:
            self.floorplan_contour.rooms = self.rooms
        
        return self.floorplan_contour, self.rooms
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for contour detection.
        
        Args:
            image: Input image
            
        Returns:
            Binary image with walls as white
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarize - walls typically appear as dark lines on light background
        if self.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to close small gaps in walls
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to thicken walls slightly for better contour detection
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    def _find_contours(self, binary: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Find contours with hierarchy information.
        
        Args:
            binary: Binary image
            
        Returns:
            Tuple of (contours list, hierarchy array)
        """
        # Find contours with full hierarchy
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None:
            hierarchy = np.array([])
        else:
            hierarchy = hierarchy[0]  # Remove extra dimension
        
        return list(contours), hierarchy
    
    def _process_contours(self, contours: List[np.ndarray], 
                         hierarchy: np.ndarray) -> List[Contour]:
        """
        Convert OpenCV contours to Contour objects with hierarchy info.
        
        Args:
            contours: List of OpenCV contours
            hierarchy: Hierarchy array from findContours
            
        Returns:
            List of Contour objects
        """
        processed = []
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            bbox = cv2.boundingRect(cnt)
            
            # Calculate centroid
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            
            # Get hierarchy info
            parent_idx = None
            children = []
            hierarchy_level = 0
            
            if len(hierarchy) > 0:
                # hierarchy[i] = [next, previous, first_child, parent]
                parent_idx = hierarchy[i][3] if hierarchy[i][3] >= 0 else None
                
                # Find children
                for j, h in enumerate(hierarchy):
                    if h[3] == i:
                        children.append(j)
                
                # Calculate hierarchy level
                level = 0
                p = parent_idx
                while p is not None and p >= 0:
                    level += 1
                    p = hierarchy[p][3] if hierarchy[p][3] >= 0 else None
                hierarchy_level = level
            
            contour_obj = Contour(
                points=cnt,
                area=area,
                perimeter=perimeter,
                bounding_box=bbox,
                centroid=(cx, cy),
                is_closed=True,
                hierarchy_level=hierarchy_level,
                parent_index=parent_idx,
                children_indices=children
            )
            
            processed.append(contour_obj)
        
        return processed
    
    def _find_floorplan_boundary(self, contours: List[Contour]) -> Optional[FloorplanContour]:
        """
        Find the outer boundary of the floorplan.
        
        Args:
            contours: List of processed contours
            
        Returns:
            FloorplanContour or None
        """
        # Filter to only outer contours (no parent)
        outer_contours = [c for c in contours 
                        if c.parent_index is None and c.area >= self.min_floorplan_area]
        
        if not outer_contours:
            # Fallback: find largest contour
            valid_contours = [c for c in contours if c.area >= self.min_floorplan_area]
            if not valid_contours:
                return None
            outer_contours = valid_contours
        
        # Select the largest outer contour as the floorplan boundary
        largest = max(outer_contours, key=lambda c: c.area)
        
        return FloorplanContour(
            contour=largest,
            total_area=largest.area,
            rooms=[]
        )
    
    def _find_rooms(self, contours: List[Contour], 
                   binary: np.ndarray) -> List[Room]:
        """
        Identify individual rooms from contours.
        
        Args:
            contours: List of processed contours
            binary: Binary image for additional analysis
            
        Returns:
            List of Room objects
        """
        rooms = []
        room_id = 0
        
        # Strategy: Look for closed contours that could represent room boundaries
        # Rooms are typically:
        # 1. Child contours of the floorplan boundary
        # 2. Have reasonable size (not too small, not too large)
        # 3. Relatively rectangular or regular shaped
        
        for contour in contours:
            # Skip if too small
            if contour.area < self.min_room_area:
                continue
            
            # Skip if it's the floorplan boundary itself
            if (self.floorplan_contour and 
                contour.area >= self.floorplan_contour.total_area * 0.9):
                continue
            
            # Check if this could be a room
            if self._is_valid_room(contour):
                room = Room(
                    id=f"room_{room_id}",
                    contour=contour,
                    room_type=self._classify_room(contour)
                )
                rooms.append(room)
                room_id += 1
        
        # Find adjacent rooms
        self._find_adjacent_rooms(rooms)
        
        return rooms
    
    def _is_valid_room(self, contour: Contour) -> bool:
        """
        Check if a contour represents a valid room.
        
        Args:
            contour: Contour to check
            
        Returns:
            True if valid room
        """
        # Check aspect ratio - rooms shouldn't be too elongated
        aspect = contour.width / max(contour.height, 1)
        if aspect > 10 or aspect < 0.1:
            return False
        
        # Check solidity (area / convex hull area) - rooms should be fairly solid
        hull = cv2.convexHull(contour.points)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = contour.area / hull_area
            if solidity < 0.3:  # Too irregular
                return False
        
        # Check if it's roughly rectangular (common for rooms)
        approx = cv2.approxPolyDP(contour.points, 0.02 * contour.perimeter, True)
        # Rooms typically have 4-8 vertices when approximated
        if len(approx) < 3 or len(approx) > 20:
            return False
        
        return True
    
    def _classify_room(self, contour: Contour) -> RoomType:
        """
        Attempt to classify room type based on characteristics.
        
        Args:
            contour: Room contour
            
        Returns:
            RoomType classification
        """
        # Basic heuristics based on size and shape
        area = contour.area
        aspect = contour.width / max(contour.height, 1)
        
        # Small, narrow spaces might be closets or hallways
        if area < self.min_room_area * 2:
            if aspect > 2 or aspect < 0.5:
                return RoomType.HALLWAY
            return RoomType.CLOSET
        
        # Very elongated spaces are likely hallways
        if aspect > 3 or aspect < 0.33:
            return RoomType.HALLWAY
        
        # TODO: More sophisticated classification using:
        # - OCR for room labels
        # - Fixture detection (toilet, sink, appliances)
        # - Connection patterns
        
        return RoomType.UNKNOWN
    
    def _find_adjacent_rooms(self, rooms: List[Room]) -> None:
        """
        Find which rooms are adjacent to each other.
        
        Args:
            rooms: List of rooms to analyze (modified in place)
        """
        adjacency_threshold = 50  # pixels
        
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if i >= j:
                    continue
                
                # Check if bounding boxes are close
                if self._are_adjacent(room1.contour, room2.contour, adjacency_threshold):
                    room1.adjacent_rooms.append(room2.id)
                    room2.adjacent_rooms.append(room1.id)
    
    def _are_adjacent(self, contour1: Contour, contour2: Contour, 
                     threshold: int) -> bool:
        """
        Check if two contours are adjacent.
        
        Args:
            contour1: First contour
            contour2: Second contour
            threshold: Maximum distance to be considered adjacent
            
        Returns:
            True if adjacent
        """
        # Quick check with bounding boxes
        x1, y1, w1, h1 = contour1.bounding_box
        x2, y2, w2, h2 = contour2.bounding_box
        
        # Expand boxes by threshold
        box1_expanded = (x1 - threshold, y1 - threshold, 
                        w1 + 2*threshold, h1 + 2*threshold)
        
        # Check if expanded box1 overlaps with box2
        if (x2 < box1_expanded[0] + box1_expanded[2] and
            x2 + w2 > box1_expanded[0] and
            y2 < box1_expanded[1] + box1_expanded[3] and
            y2 + h2 > box1_expanded[1]):
            return True
        
        return False
    
    def connect_doors_to_rooms(self, door_detections: List[Any], 
                               rooms: Optional[List[Room]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Connect detected doors to the rooms they provide access between.
        
        Args:
            door_detections: List of door Detection objects
            rooms: List of rooms (uses self.rooms if None)
            
        Returns:
            Dictionary mapping door index to room connection info
        """
        if rooms is None:
            rooms = self.rooms
        
        door_connections = {}
        
        for door_idx, door in enumerate(door_detections):
            # Get door center and bounding box
            door_center = (door.center_x, door.center_y)
            door_bbox = (door.x1, door.y1, door.x2 - door.x1, door.y2 - door.y1)
            
            connected_rooms = []
            
            for room in rooms:
                # Check if door center is inside or very close to room
                if room.contains_point(door_center):
                    connected_rooms.append(room.id)
                    continue
                
                # Check if door bounding box overlaps with room
                if self._boxes_overlap(door_bbox, room.bounding_box):
                    connected_rooms.append(room.id)
                    continue
                
                # Check if door is adjacent to room boundary (within threshold)
                if self._is_door_adjacent_to_room(door, room, threshold=30):
                    connected_rooms.append(room.id)
            
            # Determine connection type
            connection_info = {
                'door_index': door_idx,
                'door_center': door_center,
                'connected_rooms': connected_rooms,
                'connection_type': self._determine_connection_type(connected_rooms, rooms)
            }
            
            # Update room objects with door connections
            for room_id in connected_rooms:
                for room in rooms:
                    if room.id == room_id:
                        if door_idx not in room.connected_doors:
                            room.connected_doors.append(door_idx)
            
            door_connections[door_idx] = connection_info
        
        return door_connections
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple) -> bool:
        """Check if two bounding boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or
                   y1 + h1 < y2 or y2 + h2 < y1)
    
    def _is_door_adjacent_to_room(self, door: Any, room: Room, 
                                  threshold: int = 30) -> bool:
        """
        Check if a door is adjacent to a room boundary.
        
        Args:
            door: Door detection
            room: Room to check
            threshold: Maximum distance in pixels
            
        Returns:
            True if door is adjacent to room
        """
        # Sample points along door edges
        door_points = [
            (door.x1, door.center_y),  # Left edge center
            (door.x2, door.center_y),  # Right edge center
            (door.center_x, door.y1),  # Top edge center
            (door.center_x, door.y2),  # Bottom edge center
        ]
        
        for point in door_points:
            # Calculate distance from point to room contour
            dist = cv2.pointPolygonTest(room.contour.points, point, True)
            if abs(dist) <= threshold:
                return True
        
        return False
    
    def _determine_connection_type(self, connected_rooms: List[str], 
                                   all_rooms: List[Room]) -> str:
        """
        Determine what type of connection a door provides.
        
        Args:
            connected_rooms: List of connected room IDs
            all_rooms: All detected rooms
            
        Returns:
            Connection type description
        """
        num_connections = len(connected_rooms)
        
        if num_connections == 0:
            return "external"  # Door leads outside
        elif num_connections == 1:
            return "room_entry"  # Single room access
        elif num_connections == 2:
            return "room_to_room"  # Connects two rooms
        else:
            return "junction"  # Multiple room junction
    
    def visualize(self, image: np.ndarray, 
                 show_floorplan: bool = True,
                 show_rooms: bool = True,
                 show_labels: bool = True,
                 door_connections: Optional[Dict] = None) -> np.ndarray:
        """
        Visualize detected rooms and floorplan boundary.
        
        Args:
            image: Original image
            show_floorplan: Draw floorplan boundary
            show_rooms: Draw room boundaries
            show_labels: Show room labels
            door_connections: Optional door connection info to visualize
            
        Returns:
            Annotated image
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Draw floorplan boundary
        if show_floorplan and self.floorplan_contour:
            cv2.drawContours(result, [self.floorplan_contour.contour.points], 
                           -1, (0, 0, 255), 3)  # Red
        
        # Draw rooms
        if show_rooms:
            colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 255),  # Purple
                (255, 128, 0),  # Orange
            ]
            
            for i, room in enumerate(self.rooms):
                color = colors[i % len(colors)]
                cv2.drawContours(result, [room.contour.points], -1, color, 2)
                
                if show_labels:
                    # Draw room ID and type
                    label = f"{room.id}"
                    if room.room_type != RoomType.UNKNOWN:
                        label += f" ({room.room_type.value})"
                    
                    cx, cy = int(room.centroid[0]), int(room.centroid[1])
                    cv2.putText(result, label, (cx - 30, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw door connections
        if door_connections:
            for door_idx, conn in door_connections.items():
                cx, cy = int(conn['door_center'][0]), int(conn['door_center'][1])
                
                # Draw connection lines to rooms
                for room_id in conn['connected_rooms']:
                    for room in self.rooms:
                        if room.id == room_id:
                            rx, ry = int(room.centroid[0]), int(room.centroid[1])
                            cv2.line(result, (cx, cy), (rx, ry), (0, 165, 255), 1)
                
                # Label connection type
                cv2.putText(result, conn['connection_type'], (cx, cy - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return result

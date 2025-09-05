#!/usr/bin/env python3
"""
ROI Selector for Analytical Symbol Detection System
Handles region of interest selection and management
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, Dict, Any, List
try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class ROISelector:
    """Handles region of interest selection and management"""
    
    def __init__(self):
        """Initialize ROI selector"""
        self.selected_roi = None
        self.selection_active = False
        self.start_point = None
        self.end_point = None
        self.temp_roi = None
    
    def select_interactive(self, image: np.ndarray, max_width: int = 2400, 
                          max_height: int = 1400, window_title: str = "Select ROI") -> Optional[Tuple[int, int, int, int]]:
        """
        Interactive ROI selection using OpenCV
        
        Args:
            image: Input image
            max_width: Maximum window width for display
            max_height: Maximum window height for display
            window_title: Window title for selection
            
        Returns:
            ROI as (x, y, width, height) or None if cancelled
        """
        if not TKINTER_AVAILABLE:
            print("Warning: Interactive ROI selection requires tkinter, which is not available.")
            # Return a default ROI covering the center 50% of the image
            h, w = image.shape[:2]
            x, y = w//4, h//4
            roi_w, roi_h = w//2, h//2
            return (x, y, roi_w, roi_h)
            
        try:
            # Calculate scale factor for display
            height, width = image.shape[:2]
            scale = min(max_width / width, max_height / height, 1.0)
            
            # Resize image for display if needed
            if scale < 1.0:
                display_width = int(width * scale)
                display_height = int(height * scale)
                display_image = cv2.resize(image, (display_width, display_height), 
                                         interpolation=cv2.INTER_AREA)
            else:
                display_image = image.copy()
                scale = 1.0
            
            # Create window and set up ROI selection
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, display_image.shape[1], display_image.shape[0])
            
            # Use OpenCV's built-in selectROI function
            roi_display = cv2.selectROI(window_title, display_image, 
                                      showCrosshair=True, fromCenter=False)
            
            cv2.destroyAllWindows()
            
            if roi_display[2] <= 0 or roi_display[3] <= 0:
                return None
            
            # Scale ROI back to original image coordinates
            x = int(round(roi_display[0] / scale))
            y = int(round(roi_display[1] / scale))
            w = int(round(roi_display[2] / scale))
            h = int(round(roi_display[3] / scale))
            
            # Validate and clamp ROI to image bounds
            roi = self.validate_roi((x, y, w, h), image.shape[:2])
            self.selected_roi = roi
            
            return roi
            
        except Exception as e:
            print(f"Error in interactive ROI selection: {str(e)}")
            return None
    
    def select_interactive_with_callback(self, image: np.ndarray, 
                                       callback_func=None) -> Optional[Tuple[int, int, int, int]]:
        """
        Interactive ROI selection with custom mouse callback
        
        Args:
            image: Input image
            callback_func: Custom callback function for mouse events
            
        Returns:
            ROI as (x, y, width, height) or None if cancelled
        """
        self.selection_active = False
        self.start_point = None
        self.end_point = None
        self.temp_roi = None
        
        display_image = image.copy()
        clone = display_image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal display_image, clone
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.start_point = (x, y)
                self.selection_active = True
                
            elif event == cv2.EVENT_MOUSEMOVE and self.selection_active:
                display_image = clone.copy()
                cv2.rectangle(display_image, self.start_point, (x, y), (0, 255, 0), 2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                self.end_point = (x, y)
                self.selection_active = False
                
                # Calculate ROI
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                
                if roi_w > 0 and roi_h > 0:
                    self.temp_roi = (roi_x, roi_y, roi_w, roi_h)
                    cv2.rectangle(display_image, (roi_x, roi_y), 
                                (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        window_name = "Select ROI - Press 'c' to confirm, 'r' to reset, 'q' to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and self.temp_roi is not None:  # Confirm
                roi = self.validate_roi(self.temp_roi, image.shape[:2])
                self.selected_roi = roi
                cv2.destroyAllWindows()
                return roi
                
            elif key == ord('r'):  # Reset
                display_image = clone.copy()
                self.temp_roi = None
                self.start_point = None
                self.end_point = None
                self.selection_active = False
                
            elif key == ord('q') or key == 27:  # Quit
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def select_from_points(self, top_left: Tuple[int, int], 
                          bottom_right: Tuple[int, int], 
                          image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Create ROI from two points
        
        Args:
            top_left: Top-left corner (x, y)
            bottom_right: Bottom-right corner (x, y)
            image_shape: Image shape as (height, width)
            
        Returns:
            ROI as (x, y, width, height)
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        roi = self.validate_roi((x, y, w, h), image_shape)
        self.selected_roi = roi
        
        return roi
    
    def validate_roi(self, roi: Tuple[int, int, int, int], 
                    image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Validate and clamp ROI to image boundaries
        
        Args:
            roi: ROI as (x, y, width, height)
            image_shape: Image shape as (height, width)
            
        Returns:
            Validated ROI as (x, y, width, height)
        """
        x, y, w, h = roi
        img_height, img_width = image_shape
        
        # Clamp coordinates to image bounds
        x = max(0, min(img_width - 1, x))
        y = max(0, min(img_height - 1, y))
        
        # Ensure width and height are positive and within bounds
        w = max(1, min(img_width - x, w))
        h = max(1, min(img_height - y, h))
        
        return (int(x), int(y), int(w), int(h))
    
    def save_roi(self, roi: Tuple[int, int, int, int], filepath: str) -> bool:
        """
        Save ROI to JSON file
        
        Args:
            roi: ROI as (x, y, width, height)
            filepath: Path to save ROI
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            roi_data = {
                'roi': list(roi),
                'x': roi[0],
                'y': roi[1],
                'width': roi[2],
                'height': roi[3]
            }
            
            with open(filepath, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving ROI: {str(e)}")
            return False
    
    def load_roi(self, filepath: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Load ROI from JSON file
        
        Args:
            filepath: Path to ROI file
            
        Returns:
            ROI as (x, y, width, height) or None if failed
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                roi_data = json.load(f)
            
            if 'roi' in roi_data:
                roi = tuple(roi_data['roi'])
            else:
                # Legacy format
                roi = (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])
            
            self.selected_roi = roi
            return roi
            
        except Exception as e:
            print(f"Error loading ROI: {str(e)}")
            return None
    
    def crop_roi(self, image: np.ndarray, 
                roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Crop image using ROI
        
        Args:
            image: Input image
            roi: ROI to use (uses selected_roi if None)
            
        Returns:
            Cropped image or None if no valid ROI
        """
        roi_to_use = roi or self.selected_roi
        
        if roi_to_use is None:
            return None
        
        x, y, w, h = self.validate_roi(roi_to_use, image.shape[:2])
        
        try:
            cropped = image[y:y+h, x:x+w]
            return cropped
        except Exception as e:
            print(f"Error cropping ROI: {str(e)}")
            return None
    
    def visualize_roi(self, image: np.ndarray, 
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Visualize ROI on image
        
        Args:
            image: Input image
            roi: ROI to visualize (uses selected_roi if None)
            color: Rectangle color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with ROI rectangle drawn
        """
        roi_to_use = roi or self.selected_roi
        
        if roi_to_use is None:
            return image.copy()
        
        result = image.copy()
        x, y, w, h = roi_to_use
        
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Add ROI info text
        cv2.putText(result, f"ROI: {x},{y},{w},{h}", 
                   (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        return result
    
    def get_roi_info(self, roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Get information about ROI
        
        Args:
            roi: ROI to analyze (uses selected_roi if None)
            
        Returns:
            Dictionary with ROI information
        """
        roi_to_use = roi or self.selected_roi
        
        if roi_to_use is None:
            return {}
        
        x, y, w, h = roi_to_use
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'area': w * h,
            'aspect_ratio': w / h if h > 0 else 0,
            'center_x': x + w // 2,
            'center_y': y + h // 2,
            'top_left': (x, y),
            'top_right': (x + w, y),
            'bottom_left': (x, y + h),
            'bottom_right': (x + w, y + h)
        }
    
    def expand_roi(self, roi: Tuple[int, int, int, int], 
                  expansion: int, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Expand ROI by given number of pixels
        
        Args:
            roi: Original ROI
            expansion: Number of pixels to expand in each direction
            image_shape: Image shape as (height, width)
            
        Returns:
            Expanded ROI
        """
        x, y, w, h = roi
        
        new_x = x - expansion
        new_y = y - expansion
        new_w = w + 2 * expansion
        new_h = h + 2 * expansion
        
        expanded_roi = self.validate_roi((new_x, new_y, new_w, new_h), image_shape)
        return expanded_roi
    
    def shrink_roi(self, roi: Tuple[int, int, int, int], 
                  shrinkage: int) -> Tuple[int, int, int, int]:
        """
        Shrink ROI by given number of pixels
        
        Args:
            roi: Original ROI
            shrinkage: Number of pixels to shrink from each direction
            
        Returns:
            Shrunk ROI
        """
        x, y, w, h = roi
        
        new_x = x + shrinkage
        new_y = y + shrinkage
        new_w = max(1, w - 2 * shrinkage)
        new_h = max(1, h - 2 * shrinkage)
        
        return (int(new_x), int(new_y), int(new_w), int(new_h))
    
    def center_roi(self, roi: Tuple[int, int, int, int], 
                  image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Center ROI in image
        
        Args:
            roi: Original ROI
            image_shape: Image shape as (height, width)
            
        Returns:
            Centered ROI
        """
        _, _, w, h = roi
        img_height, img_width = image_shape
        
        new_x = (img_width - w) // 2
        new_y = (img_height - h) // 2
        
        centered_roi = self.validate_roi((new_x, new_y, w, h), image_shape)
        return centered_roi
    
    def get_multiple_rois_interactive(self, image: np.ndarray, 
                                    num_rois: int = 1) -> List[Tuple[int, int, int, int]]:
        """
        Select multiple ROIs interactively
        
        Args:
            image: Input image
            num_rois: Number of ROIs to select
            
        Returns:
            List of ROIs
        """
        rois = []
        
        for i in range(num_rois):
            print(f"Select ROI {i+1} of {num_rois}")
            roi = self.select_interactive(image, window_title=f"Select ROI {i+1}/{num_rois}")
            
            if roi is not None:
                rois.append(roi)
            else:
                print(f"ROI selection {i+1} cancelled")
                break
        
        return rois
    
    def reset_selection(self):
        """Reset current ROI selection"""
        self.selected_roi = None
        self.selection_active = False
        self.start_point = None
        self.end_point = None
        self.temp_roi = None
    
    def extract_roi(self, image: np.ndarray, 
                   roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract ROI from image (simple method for compatibility)"""
        x, y, w, h = self.validate_roi(roi, image.shape[:2])
        return image[y:y+h, x:x+w].copy()
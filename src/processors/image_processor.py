#!/usr/bin/env python3
"""
Image Processor for Analytical Symbol Detection System
Handles image preprocessing operations like grayscale conversion, edge detection, etc.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(self):
        """Initialize image preprocessor"""
        pass
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    
    def extract_edges(self, image: np.ndarray, low_threshold: int = 50, 
                     high_threshold: int = 150, blur_kernel: int = 3) -> np.ndarray:
        """
        Extract edges using Canny edge detection
        
        Args:
            image: Input image
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            blur_kernel: Gaussian blur kernel size
            
        Returns:
            Binary edge image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if blur_kernel > 0:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        return edges
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        
        Args:
            image: Input image
            clip_limit: Clipping limit for CLAHE
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def binarize(self, image: np.ndarray, method: str = 'otsu', 
                threshold_value: int = 127, adaptive_block_size: int = 11,
                adaptive_c: int = 2) -> np.ndarray:
        """
        Binarize image using various methods
        
        Args:
            image: Input image
            method: Binarization method ('otsu', 'adaptive', 'simple')
            threshold_value: Threshold value for simple thresholding
            adaptive_block_size: Block size for adaptive thresholding
            adaptive_c: Constant subtracted from mean in adaptive thresholding
            
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                adaptive_block_size, adaptive_c
            )
        elif method == 'simple':
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown binarization method: {method}")
        
        return binary
    
    def apply_morphology(self, image: np.ndarray, operation: str,
                        kernel_shape: str = 'rect', kernel_size: Tuple[int, int] = (3, 3),
                        iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations
        
        Args:
            image: Input binary image
            operation: Morphological operation ('open', 'close', 'erode', 'dilate', 'gradient', 'tophat', 'blackhat')
            kernel_shape: Kernel shape ('rect', 'ellipse', 'cross')
            kernel_size: Kernel size
            iterations: Number of iterations
            
        Returns:
            Processed image
        """
        # Create kernel
        if kernel_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        elif kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        else:
            raise ValueError(f"Unknown kernel shape: {kernel_shape}")
        
        # Apply morphological operation
        operation_map = {
            'erode': cv2.MORPH_ERODE,
            'dilate': cv2.MORPH_DILATE,
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        result = cv2.morphologyEx(image, operation_map[operation], kernel, iterations=iterations)
        
        return result
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, 
                        beta: int = 0) -> np.ndarray:
        """
        Enhance contrast using linear transformation
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Contrast-enhanced image
        """
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian',
                    kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input image
            method: Denoising method ('gaussian', 'median', 'bilateral', 'nlmeans')
            kernel_size: Kernel size for filtering
            sigma: Standard deviation for Gaussian filtering
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == 'median':
            denoised = cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            if len(image.shape) == 3:
                denoised = cv2.bilateralFilter(image, kernel_size, 80, 80)
            else:
                # Convert to 3-channel for bilateral filtering
                temp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                temp = cv2.bilateralFilter(temp, kernel_size, 80, 80)
                denoised = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        elif method == 'nlmeans':
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return denoised
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None,
                    scale_factor: float = None, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image
        
        Args:
            image: Input image
            size: Target size as (width, height)
            scale_factor: Scale factor for resizing
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        if size is not None:
            resized = cv2.resize(image, size, interpolation=interpolation)
        elif scale_factor is not None:
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            resized = cv2.resize(image, (width, height), interpolation=interpolation)
        else:
            raise ValueError("Either size or scale_factor must be provided")
        
        return resized
    
    def rotate_image(self, image: np.ndarray, angle: float, 
                    scale: float = 1.0, border_mode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
        """
        Rotate image by given angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            scale: Scaling factor
            border_mode: Border handling mode
            
        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Calculate new image dimensions to contain entire rotated image
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust translation to center the image
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                               flags=cv2.INTER_LINEAR, borderMode=border_mode)
        
        return rotated
    
    def crop_image(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image to region of interest
        
        Args:
            image: Input image
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Cropped image
        """
        x, y, w, h = roi
        
        # Ensure ROI is within image bounds
        height, width = image.shape[:2]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def pad_image(self, image: np.ndarray, padding: Tuple[int, int, int, int],
                 border_type: int = cv2.BORDER_CONSTANT, value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Add padding to image
        
        Args:
            image: Input image
            padding: Padding as (top, bottom, left, right)
            border_type: Border type for padding
            value: Value for constant border
            
        Returns:
            Padded image
        """
        top, bottom, left, right = padding
        
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                  border_type, value=value)
        
        return padded
    
    def get_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get image statistics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': int(np.min(image)),
            'max': int(np.max(image)),
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'median': float(np.median(gray))
        }
        
        return stats
    
    def save_preview_images(self, image: np.ndarray, output_dir: str, 
                          base_name: str = "page") -> Dict[str, str]:
        """
        Save preview images in different processing stages
        
        Args:
            image: Input image
            output_dir: Output directory
            base_name: Base name for files
            
        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Original/grayscale
        if len(image.shape) == 3:
            gray = self.convert_to_grayscale(image)
        else:
            gray = image.copy()
        
        gray_path = os.path.join(output_dir, f"{base_name}_gray.png")
        cv2.imwrite(gray_path, gray)
        saved_files['grayscale'] = gray_path
        
        # Edges
        edges = self.extract_edges(gray)
        edges_path = os.path.join(output_dir, f"{base_name}_edges.png")
        cv2.imwrite(edges_path, edges)
        saved_files['edges'] = edges_path
        
        # Enhanced contrast
        enhanced = self.apply_clahe(gray)
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        cv2.imwrite(enhanced_path, enhanced)
        saved_files['enhanced'] = enhanced_path
        
        # Binary
        binary = self.binarize(gray, method='otsu')
        binary_path = os.path.join(output_dir, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary)
        saved_files['binary'] = binary_path
        
        return saved_files
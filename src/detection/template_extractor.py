#!/usr/bin/env python3
"""
Template Extractor for Analytical Symbol Detection System
Handles template extraction and analysis from ROI regions
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os


class TemplateExtractor:
    """Handles template extraction and analysis from ROI regions"""
    
    def __init__(self):
        """Initialize template extractor"""
        pass
    
    def extract_template(self, image: np.ndarray, 
                        roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract template from image using ROI
        
        Args:
            image: Source image
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Extracted template image
        """
        x, y, w, h = roi
        
        # Ensure ROI is within image bounds
        img_height, img_width = image.shape[:2]
        x = max(0, min(img_width - 1, x))
        y = max(0, min(img_height - 1, y))
        w = max(1, min(img_width - x, w))
        h = max(1, min(img_height - y, h))
        
        # Extract template
        template = image[y:y+h, x:x+w].copy()
        return template
    
    def create_interior_mask(self, template: np.ndarray,
                           use_otsu: bool = True,
                           thresh_val: int = 127,
                           invert: bool = True,
                           close_ksize: int = 3,
                           min_area_ratio: float = 0.003,
                           keep_mode: str = "largest",
                           center_ratio: float = 0.85) -> np.ndarray:
        """
        Build a mask of all enclosed regions (holes) inside the symbol.
        Uses cv2.RETR_CCOMP so child contours are holes (parent != -1).
        
        Args:
            template: Template image
            use_otsu: Use Otsu binarization (adapts threshold)
            thresh_val: Threshold value if not using Otsu
            invert: Set True when strokes are dark; we invert to make strokes=255
            close_ksize: Morphology close kernel (>=1) to seal tiny gaps in strokes
            min_area_ratio: Drop tiny enclosed regions by area relative to ROI
            keep_mode: "all", "largest", or "centered"
            center_ratio: For "centered": keep CCs whose centroid lies in central box
            
        Returns:
            Binary mask of interior regions
        """
        # Ensure single channel
        if template.ndim == 3:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template.copy()
        
        # Binarize: strokes -> 255
        if use_otsu:
            _, bw = cv2.threshold(gray, 0, 255,
                                  (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(gray, thresh_val, 255,
                                  cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
        
        # Seal small gaps so holes are truly enclosed
        if close_ksize and close_ksize > 0:
            k = np.ones((close_ksize, close_ksize), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
        
        # Find contours with two-level hierarchy (components + holes)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape
        mask = np.zeros((H, W), np.uint8)
        
        if hierarchy is None or len(contours) == 0:
            return mask
        
        # Center box for the "centered" filter
        if keep_mode == "centered":
            cx1 = int((1 - center_ratio) / 2 * W)
            cy1 = int((1 - center_ratio) / 2 * H)
            cx2 = W - cx1
            cy2 = H - cy1
        
        # Draw all hole contours (parent != -1) that pass filters
        roi_area = float(H * W)
        keep_idxs = []
        
        for i, cnt in enumerate(contours):
            parent = hierarchy[0][i][3]  # -1 = no parent (outer); >=0 = hole
            if parent == -1:
                continue  # skip outer strokes
            
            area = cv2.contourArea(cnt)
            if area < min_area_ratio * roi_area:
                continue  # tiny specks
            
            if keep_mode == "centered":
                m = cv2.moments(cnt)
                if m["m00"] == 0:
                    continue
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                if not (cx1 <= cx <= cx2 and cy1 <= cy <= cy2):
                    continue
            
            keep_idxs.append(i)
        
        # Draw union
        for i in keep_idxs:
            cv2.drawContours(mask, contours, i, 255, thickness=-1)
        
        # Optionally keep only the largest connected component
        if keep_mode == "largest":
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                ii = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = np.where(labels == ii, 255, 0).astype(np.uint8)
        
        # Final clean
        if close_ksize and close_ksize > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        
        return mask
    
    def tighten_roi(self, roi: Tuple[int, int, int, int], 
                   mask: np.ndarray,
                   full_image_shape: Tuple[int, int],
                   pad_px: int = 2,
                   pad_ratio: float = 0.03,
                   min_size: int = 8) -> Tuple[int, int, int, int]:
        """
        Given an initial ROI and a binary mask created from that ROI crop,
        return a tightened ROI that is the axis-aligned bounding box of the mask.
        
        Args:
            roi: Initial ROI as (x, y, width, height)
            mask: Binary mask from the ROI crop
            full_image_shape: Full image shape as (height, width)
            pad_px: Pixel padding
            pad_ratio: Relative padding to bbox size
            min_size: Minimum size constraint
            
        Returns:
            Tightened ROI as (x, y, width, height)
        """
        x, y, w, h = roi
        full_H, full_W = full_image_shape
        
        if mask.ndim == 3:
            mask = mask[..., 0]
        
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            # No foreground in mask -> keep original ROI
            return roi
        
        # bbox in *ROI coordinates*
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max())
        y1 = int(ys.max())
        
        # padding (pixel + relative to bbox)
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        pad = int(round(pad_px + pad_ratio * max(bw, bh)))
        
        # map to *page coordinates* + pad + clamp
        nx1 = max(0, x + x0 - pad)
        ny1 = max(0, y + y0 - pad)
        nx2 = min(full_W - 1, x + x1 + pad)
        ny2 = min(full_H - 1, y + y1 + pad)
        
        nw = max(min_size, nx2 - nx1 + 1)
        nh = max(min_size, ny2 - ny1 + 1)
        
        return (int(nx1), int(ny1), int(nw), int(nh))
    
    def analyze_template_features(self, template: np.ndarray) -> Dict[str, Any]:
        """
        Analyze template features for quality assessment
        
        Args:
            template: Template image
            
        Returns:
            Dictionary with template features
        """
        if template.ndim == 3:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray = template.copy()
        
        features = {}
        
        # Basic statistics
        features['mean'] = float(np.mean(gray))
        features['std'] = float(np.std(gray))
        features['min'] = int(np.min(gray))
        features['max'] = int(np.max(gray))
        features['shape'] = template.shape
        features['area'] = template.shape[0] * template.shape[1]
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        features['edge_density'] = float(edge_pixels / total_pixels)
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['num_contours'] = len(contours)
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features['largest_contour_area'] = float(max(areas))
            features['total_contour_area'] = float(sum(areas))
            features['avg_contour_area'] = float(np.mean(areas))
        else:
            features['largest_contour_area'] = 0.0
            features['total_contour_area'] = 0.0
            features['avg_contour_area'] = 0.0
        
        # Texture analysis using local binary patterns (simplified)
        try:
            # Calculate texture uniformity
            gray_normalized = gray.astype(np.float32) / 255.0
            laplacian = cv2.Laplacian(gray_normalized, cv2.CV_32F)
            features['texture_variance'] = float(np.var(laplacian))
        except:
            features['texture_variance'] = 0.0
        
        # Template quality score (heuristic)
        quality_score = 0.0
        
        # Higher score for good edge density (not too sparse, not too dense)
        if 0.02 < features['edge_density'] < 0.3:
            quality_score += 0.3
        
        # Higher score for reasonable contrast
        if features['std'] > 20:
            quality_score += 0.3
        
        # Higher score for reasonable size
        if 20 <= min(template.shape[:2]) <= 200:
            quality_score += 0.2
        
        # Higher score for reasonable number of contours
        if 1 <= features['num_contours'] <= 10:
            quality_score += 0.2
        
        features['quality_score'] = quality_score
        
        return features
    
    def preprocess_template(self, template: np.ndarray, 
                          use_edges: bool = True,
                          enhance_contrast: bool = True,
                          denoise: bool = True) -> np.ndarray:
        """
        Preprocess template for better matching
        
        Args:
            template: Input template
            use_edges: Convert to edge representation
            enhance_contrast: Enhance contrast using CLAHE
            denoise: Apply denoising
            
        Returns:
            Preprocessed template
        """
        # Convert to grayscale if needed
        if template.ndim == 3:
            processed = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            processed = template.copy()
        
        # Denoise
        if denoise:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Enhance contrast
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        # Convert to edges if requested
        if use_edges:
            processed = cv2.Canny(processed, 50, 150)
        
        return processed
    
    def save_template(self, template: np.ndarray, filepath: str) -> bool:
        """
        Save template to file
        
        Args:
            template: Template image
            filepath: Path to save template
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            cv2.imwrite(filepath, template)
            return True
        except Exception as e:
            print(f"Error saving template: {str(e)}")
            return False
    
    def load_template(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load template from file
        
        Args:
            filepath: Path to template file
            
        Returns:
            Template image or None if failed
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            template = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            return template
        except Exception as e:
            print(f"Error loading template: {str(e)}")
            return None
    
    def create_multi_scale_templates(self, template: np.ndarray, 
                                   scales: list = [0.8, 0.9, 1.0, 1.1, 1.2]) -> Dict[float, np.ndarray]:
        """
        Create templates at multiple scales
        
        Args:
            template: Base template
            scales: List of scale factors
            
        Returns:
            Dictionary mapping scale to template
        """
        templates = {}
        
        for scale in scales:
            if scale == 1.0:
                templates[scale] = template.copy()
            else:
                height, width = template.shape[:2]
                new_height = int(height * scale)
                new_width = int(width * scale)
                
                if new_height > 0 and new_width > 0:
                    scaled_template = cv2.resize(template, (new_width, new_height), 
                                               interpolation=cv2.INTER_LINEAR)
                    templates[scale] = scaled_template
        
        return templates
    
    def create_rotated_templates(self, template: np.ndarray, 
                               angles: list = [0, 90, 180, 270]) -> Dict[float, np.ndarray]:
        """
        Create templates at multiple rotation angles
        
        Args:
            template: Base template
            angles: List of rotation angles in degrees
            
        Returns:
            Dictionary mapping angle to template
        """
        templates = {}
        
        for angle in angles:
            if angle == 0:
                templates[angle] = template.copy()
            else:
                rotated_template = self.rotate_template(template, angle)
                templates[angle] = rotated_template
        
        return templates
    
    def rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate template by given angle while keeping all content
        
        Args:
            template: Input template
            angle: Rotation angle in degrees
            
        Returns:
            Rotated template
        """
        (h, w) = template.shape[:2]
        cX, cY = w // 2, h // 2
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        # Adjust translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        # Perform rotation
        rotated = cv2.warpAffine(template, M, (nW, nH), 
                               flags=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def extract_and_process_template(self, image: np.ndarray, 
                                   roi: Tuple[int, int, int, int],
                                   use_interior_mask: bool = True,
                                   tighten_roi_bounds: bool = True,
                                   preprocess: bool = True,
                                   use_edges: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete template extraction and processing pipeline
        
        Args:
            image: Source image
            roi: Initial ROI
            use_interior_mask: Apply interior mask for tightening
            tighten_roi_bounds: Tighten ROI based on mask
            preprocess: Apply preprocessing
            use_edges: Use edge representation
            
        Returns:
            Tuple of (processed_template, metadata)
        """
        metadata = {'original_roi': roi}
        
        # Extract initial template
        template = self.extract_template(image, roi)
        metadata['original_template_shape'] = template.shape
        
        # Create interior mask if requested
        if use_interior_mask:
            mask = self.create_interior_mask(template)
            metadata['interior_mask_created'] = True
            
            # Tighten ROI if requested
            if tighten_roi_bounds:
                tightened_roi = self.tighten_roi(roi, mask, image.shape[:2])
                metadata['tightened_roi'] = tightened_roi
                
                # Re-extract with tightened ROI
                template = self.extract_template(image, tightened_roi)
                metadata['final_template_shape'] = template.shape
        
        # Preprocess template
        if preprocess:
            template = self.preprocess_template(template, use_edges=use_edges)
            metadata['preprocessed'] = True
        
        # Analyze features
        features = self.analyze_template_features(template)
        metadata['features'] = features
        
        return template, metadata
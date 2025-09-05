#!/usr/bin/env python3
"""
PDF Processor for Analytical Symbol Detection System
Handles PDF to image conversion using PyMuPDF (fitz)
"""

import os
import tempfile
from typing import Tuple, Optional
import numpy as np
import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFProcessor:
    """Handles PDF file operations and conversion to images"""
    
    def __init__(self, pdf_path: str, page: int = 0, zoom: float = 6.0):
        """
        Initialize PDF processor
        
        Args:
            pdf_path: Path to PDF file
            page: Page index (0-based)
            zoom: Zoom factor for rendering
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) not found. Install with: pip install pymupdf")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.page = page
        self.zoom = zoom
        self._doc = None
        
        # Validate page index
        if page < 0 or page >= self.get_page_count():
            raise IndexError(f"Page index {page} out of range for PDF with {self.get_page_count()} pages")
    
    def __enter__(self):
        """Context manager entry"""
        self._doc = fitz.open(self.pdf_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._doc:
            self._doc.close()
            self._doc = None
    
    def get_page_count(self) -> int:
        """Get total number of pages in PDF"""
        with fitz.open(self.pdf_path) as doc:
            return len(doc)
    
    def get_page_dimensions(self, page_index: Optional[int] = None) -> Tuple[float, float]:
        """
        Get dimensions of a specific page
        
        Args:
            page_index: Page index (defaults to current page)
            
        Returns:
            Tuple of (width, height) in points
        """
        page_idx = page_index if page_index is not None else self.page
        
        with fitz.open(self.pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"Page index {page_idx} out of range")
            
            page = doc.load_page(page_idx)
            rect = page.rect
            return rect.width, rect.height
    
    def render_to_png(self, output_path: str, page_index: Optional[int] = None, 
                      zoom_factor: Optional[float] = None) -> str:
        """
        Render PDF page to PNG file
        
        Args:
            output_path: Path where PNG should be saved
            page_index: Page index (defaults to current page)
            zoom_factor: Zoom factor (defaults to current zoom)
            
        Returns:
            Path to saved PNG file
        """
        page_idx = page_index if page_index is not None else self.page
        zoom_val = zoom_factor if zoom_factor is not None else self.zoom
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with fitz.open(self.pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"Page index {page_idx} out of range")
            
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(zoom_val, zoom_val)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(output_path)
        
        return output_path
    
    def render_to_array(self, page_index: Optional[int] = None, 
                       zoom_factor: Optional[float] = None) -> np.ndarray:
        """
        Render PDF page directly to numpy array
        
        Args:
            page_index: Page index (defaults to current page)
            zoom_factor: Zoom factor (defaults to current zoom)
            
        Returns:
            BGR image as numpy array
        """
        page_idx = page_index if page_index is not None else self.page
        zoom_val = zoom_factor if zoom_factor is not None else self.zoom
        
        with fitz.open(self.pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"Page index {page_idx} out of range")
            
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(zoom_val, zoom_val)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_data = pix.samples
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Convert from RGB to BGR for OpenCV compatibility
            if pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            return img
    
    def render_to_temp_file(self, page_index: Optional[int] = None, 
                           zoom_factor: Optional[float] = None, 
                           suffix: str = '.png') -> str:
        """
        Render PDF page to temporary file
        
        Args:
            page_index: Page index (defaults to current page)
            zoom_factor: Zoom factor (defaults to current zoom)
            suffix: File suffix for temporary file
            
        Returns:
            Path to temporary PNG file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
        
        return self.render_to_png(temp_path, page_index, zoom_factor)
    
    def get_page_info(self, page_index: Optional[int] = None) -> dict:
        """
        Get detailed information about a page
        
        Args:
            page_index: Page index (defaults to current page)
            
        Returns:
            Dictionary with page information
        """
        page_idx = page_index if page_index is not None else self.page
        
        with fitz.open(self.pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"Page index {page_idx} out of range")
            
            page = doc.load_page(page_idx)
            
            return {
                'page_number': page_idx + 1,
                'width': page.rect.width,
                'height': page.rect.height,
                'rotation': page.rotation,
                'media_box': {
                    'x0': page.mediabox.x0,
                    'y0': page.mediabox.y0,
                    'x1': page.mediabox.x1,
                    'y1': page.mediabox.y1
                }
            }
    
    def extract_text(self, page_index: Optional[int] = None) -> str:
        """
        Extract text from PDF page
        
        Args:
            page_index: Page index (defaults to current page)
            
        Returns:
            Extracted text as string
        """
        page_idx = page_index if page_index is not None else self.page
        
        with fitz.open(self.pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                raise IndexError(f"Page index {page_idx} out of range")
            
            page = doc.load_page(page_idx)
            return page.get_text()
    
    def get_pdf_metadata(self) -> dict:
        """
        Get PDF metadata
        
        Returns:
            Dictionary with PDF metadata
        """
        with fitz.open(self.pdf_path) as doc:
            metadata = doc.metadata
            metadata['page_count'] = len(doc)
            return metadata
    
    def validate_pdf(self) -> bool:
        """
        Validate that PDF file can be opened and processed
        
        Returns:
            True if PDF is valid and readable
        """
        try:
            with fitz.open(self.pdf_path) as doc:
                if len(doc) == 0:
                    return False
                
                # Try to load first page
                page = doc.load_page(0)
                return page is not None
        except Exception:
            return False
    
    def create_preview_image(self, output_path: str, max_width: int = 800, 
                           max_height: int = 600) -> str:
        """
        Create a preview image with constrained dimensions
        
        Args:
            output_path: Path where preview should be saved
            max_width: Maximum width for preview
            max_height: Maximum height for preview
            
        Returns:
            Path to saved preview image
        """
        # Get page dimensions
        page_width, page_height = self.get_page_dimensions()
        
        # Calculate zoom to fit within constraints
        width_ratio = max_width / page_width
        height_ratio = max_height / page_height
        preview_zoom = min(width_ratio, height_ratio, 1.0)  # Don't upscale
        
        return self.render_to_png(output_path, zoom_factor=preview_zoom)
    
    @staticmethod
    def ensure_image_from_pdf(pdf_path: str, page: int, zoom: float, 
                             output_dir: str) -> str:
        """
        Utility function to ensure image exists, creating it if necessary
        
        Args:
            pdf_path: Path to PDF file
            page: Page index
            zoom: Zoom factor
            output_dir: Directory for output image
            
        Returns:
            Path to PNG image file
        """
        output_path = os.path.join(output_dir, f"page{page}.png")
        
        if not os.path.exists(output_path):
            processor = PDFProcessor(pdf_path, page, zoom)
            processor.render_to_png(output_path)
            print(f"[i] Rendered PDF → {output_path} (zoom={zoom})")
        else:
            print(f"[i] Using cached image: {output_path}")
        
        return output_path
# Phase 5: Streamlit User Interface

## Objective
Implement a comprehensive Streamlit dashboard that supports all argparse parameters and provides an intuitive interface for symbol detection.

## Tasks

### Task 5.1: Main Streamlit Application
**File**: `src/ui/app.py`

**Implementation**:
```python
import streamlit as st
import os
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

# Import core components
from src.core.config_manager import ConfigurationManager
from src.processors.pdf_processor import PDFProcessor
from src.processors.image_processor import ImagePreprocessor
from src.processors.roi_selector import ROISelector
from src.detection.template_extractor import TemplateExtractor
from src.detection.detection_pipeline import DetectionPipeline
from src.ui.components import (
    sidebar_config,
    roi_selection_component,
    detection_results_component,
    visualization_component
)

class SymbolDetectionApp:
    """Main Streamlit application for symbol detection"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Analytical Symbol Detection",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'config' not in st.session_state:
            st.session_state.config = self.config_manager.config
        
        if 'pdf_loaded' not in st.session_state:
            st.session_state.pdf_loaded = False
            
        if 'roi_selected' not in st.session_state:
            st.session_state.roi_selected = False
            
        if 'template_extracted' not in st.session_state:
            st.session_state.template_extracted = False
            
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
            
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
            
        if 'current_template' not in st.session_state:
            st.session_state.current_template = None
    
    def run(self):
        """Main application loop"""
        st.title("🔍 Analytical Symbol Detection System")
        st.markdown("---")
        
        # Sidebar configuration
        with st.sidebar:
            self.render_sidebar()
        
        # Main content area
        if st.session_state.pdf_loaded:
            self.render_main_content()
        else:
            self.render_welcome_screen()
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.header("Configuration")
        
        # File upload section
        st.subheader("📄 PDF Input")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Select a floor plan PDF to analyze"
        )
        
        if uploaded_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.config.pdf_path = tmp_file.name
                st.session_state.pdf_loaded = True
        
        if st.session_state.pdf_loaded:
            # PDF settings
            st.subheader("⚙️ PDF Settings")
            
            processor = PDFProcessor(st.session_state.config.pdf_path)
            total_pages = processor.get_page_count()
            
            st.session_state.config.page = st.number_input(
                "Page Number",
                min_value=0,
                max_value=total_pages - 1,
                value=st.session_state.config.page,
                help=f"Select page (0-{total_pages-1})"
            )
            
            st.session_state.config.zoom = st.slider(
                "Zoom Factor",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.config.zoom,
                step=0.5,
                help="Higher zoom = better quality but larger file"
            )
            
            # Detection parameters
            st.subheader("🎯 Detection Parameters")
            
            st.session_state.config.threshold = st.slider(
                "Detection Threshold",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.config.threshold,
                step=0.05,
                help="Lower = more detections (may include false positives)"
            )
            
            # Scales
            scales_str = st.text_input(
                "Search Scales",
                value=",".join(map(str, st.session_state.config.scales)),
                help="Comma-separated scale factors (e.g., 0.9,1.0,1.1)"
            )
            st.session_state.config.scales = [
                float(s.strip()) for s in scales_str.split(',') if s.strip()
            ]
            
            # Angles
            angles_str = st.text_input(
                "Search Angles (degrees)",
                value=",".join(map(str, st.session_state.config.angles)),
                help="Comma-separated angles (e.g., 0,90,180,270)"
            )
            st.session_state.config.angles = [
                float(a.strip()) for a in angles_str.split(',') if a.strip()
            ]
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                st.session_state.config.method = st.selectbox(
                    "Matching Method",
                    options=['CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'],
                    index=0,
                    help="Template matching algorithm"
                )
                
                st.session_state.config.use_edges = st.checkbox(
                    "Use Edge Detection",
                    value=st.session_state.config.use_edges,
                    help="Match on edges instead of grayscale"
                )
                
                st.session_state.config.coarse_scale = st.slider(
                    "Coarse Search Scale",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.config.coarse_scale,
                    step=0.1,
                    help="Downscale factor for initial search"
                )
                
                st.session_state.config.topk = st.number_input(
                    "Top-K Candidates",
                    min_value=10,
                    max_value=1000,
                    value=st.session_state.config.topk,
                    step=10,
                    help="Number of coarse candidates to refine"
                )
                
                st.session_state.config.refine_pad = st.slider(
                    "Refinement Padding",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.config.refine_pad,
                    step=0.1,
                    help="Padding ratio for refinement stage"
                )
            
            # Symbol class selection
            st.subheader("🏷️ Symbol Class")
            st.session_state.config.class_name = st.selectbox(
                "Symbol Type",
                options=['hvac', 'door', 'window', 'electrical', 'plumbing', 'custom'],
                help="Select the type of symbol to detect"
            )
            
            # Action buttons
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Config", use_container_width=True):
                    self.save_configuration()
            
            with col2:
                if st.button("📂 Load Config", use_container_width=True):
                    self.load_configuration()
    
    def render_main_content(self):
        """Render main content area"""
        # Create tabs for different stages
        tab1, tab2, tab3, tab4 = st.tabs([
            "📷 Image Processing",
            "✂️ ROI Selection", 
            "🔍 Detection",
            "📊 Results"
        ])
        
        with tab1:
            self.render_image_processing_tab()
        
        with tab2:
            self.render_roi_selection_tab()
        
        with tab3:
            self.render_detection_tab()
        
        with tab4:
            self.render_results_tab()
    
    def render_image_processing_tab(self):
        """Render image processing tab"""
        st.header("Image Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🖼️ Render PDF Page", type="primary", use_container_width=True):
                with st.spinner("Rendering PDF..."):
                    self.render_pdf_page()
        
        with col2:
            if st.session_state.current_image is not None:
                st.success("✅ Page rendered successfully")
        
        if st.session_state.current_image is not None:
            st.subheader("Rendered Page")
            
            # Display options
            display_mode = st.radio(
                "Display Mode",
                options=['Original', 'Grayscale', 'Edges', 'Binary'],
                horizontal=True
            )
            
            processor = ImagePreprocessor()
            display_image = st.session_state.current_image.copy()
            
            if display_mode == 'Grayscale':
                display_image = processor.convert_to_grayscale(display_image)
            elif display_mode == 'Edges':
                display_image = processor.extract_edges(display_image)
            elif display_mode == 'Binary':
                display_image = processor.binarize(display_image)
            
            # Convert to RGB for display
            if len(display_image.shape) == 2:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
            else:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            st.image(display_image, use_column_width=True)
            
            # Image info
            h, w = st.session_state.current_image.shape[:2]
            st.info(f"Image size: {w} × {h} pixels")
    
    def render_roi_selection_tab(self):
        """Render ROI selection tab"""
        st.header("ROI Selection")
        
        if st.session_state.current_image is None:
            st.warning("⚠️ Please render a PDF page first")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✂️ Select ROI Interactively", use_container_width=True):
                st.info("Please use the popup window to select ROI")
                # Note: In production, would integrate with canvas widget
        
        with col2:
            if st.button("📁 Load Saved ROI", use_container_width=True):
                self.load_roi()
        
        with col3:
            manual_roi = st.checkbox("Enter ROI manually")
        
        if manual_roi:
            st.subheader("Manual ROI Entry")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x = st.number_input("X", min_value=0, value=0)
            with col2:
                y = st.number_input("Y", min_value=0, value=0)
            with col3:
                w = st.number_input("Width", min_value=1, value=100)
            with col4:
                h = st.number_input("Height", min_value=1, value=100)
            
            if st.button("Apply Manual ROI"):
                st.session_state.config.roi = (x, y, w, h)
                st.session_state.roi_selected = True
                self.extract_template()
        
        # Display template if extracted
        if st.session_state.template_extracted and st.session_state.current_template is not None:
            st.subheader("Extracted Template")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(
                    cv2.cvtColor(st.session_state.current_template, cv2.COLOR_BGR2RGB),
                    caption="Template",
                    use_column_width=True
                )
            
            with col2:
                # Template analysis
                extractor = TemplateExtractor()
                features = extractor.analyze_template_features(st.session_state.current_template)
                
                st.write("**Template Features:**")
                st.write(f"- Mean intensity: {features['mean']:.2f}")
                st.write(f"- Edge density: {features['edge_density']:.3f}")
                st.write(f"- Contours: {features['num_contours']}")
    
    def render_detection_tab(self):
        """Render detection tab"""
        st.header("Symbol Detection")
        
        if not st.session_state.template_extracted:
            st.warning("⚠️ Please select ROI and extract template first")
            return
        
        # Detection controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                with st.spinner("Detecting symbols..."):
                    self.run_detection()
        
        with col2:
            if st.session_state.detection_results:
                num_detections = len(st.session_state.detection_results.detections)
                st.success(f"✅ Found {num_detections} symbols")
        
        # Progress indicator
        if st.session_state.detection_results:
            st.progress(1.0)
            
            # Detection preview
            st.subheader("Detection Preview")
            self.render_detection_preview()
    
    def render_results_tab(self):
        """Render results tab"""
        st.header("Detection Results")
        
        if not st.session_state.detection_results:
            st.info("No detection results available. Run detection first.")
            return
        
        results = st.session_state.detection_results
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(results.detections))
        
        with col2:
            avg_conf = np.mean([d.confidence for d in results.detections])
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        with col3:
            st.metric("Processing Time", f"{results.processing_time:.2f}s")
        
        with col4:
            st.metric("Symbol Class", st.session_state.config.class_name)
        
        # Export options
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export YOLO Labels", use_container_width=True):
                self.export_yolo_labels()
        
        with col2:
            if st.button("📊 Export CSV", use_container_width=True):
                self.export_csv()
        
        with col3:
            if st.button("🖼️ Save Annotated Image", use_container_width=True):
                self.save_annotated_image()
        
        # Detailed results table
        st.subheader("Detection Details")
        
        if st.checkbox("Show detailed results"):
            self.render_results_table()
    
    def render_welcome_screen(self):
        """Render welcome screen when no PDF is loaded"""
        st.info("👋 Welcome to the Analytical Symbol Detection System!")
        
        st.markdown("""
        ### Getting Started
        
        1. **Upload a PDF** - Use the sidebar to upload a floor plan PDF
        2. **Configure Settings** - Adjust detection parameters as needed
        3. **Select ROI** - Choose a region containing your target symbol
        4. **Run Detection** - Find all instances of the symbol
        5. **Export Results** - Save detections in various formats
        
        ### Features
        
        - 🔄 Multi-scale and multi-angle detection
        - ⚡ Coarse-to-fine search strategy
        - 🎯 Symbol-specific detectors
        - 📊 YOLO format export
        - 🖼️ Visual results with annotations
        
        ### Supported Symbol Types
        
        - HVAC symbols
        - Doors
        - Windows  
        - Electrical components
        - Plumbing fixtures
        - Custom symbols
        """)
    
    # Helper methods
    def render_pdf_page(self):
        """Render current PDF page"""
        processor = PDFProcessor(
            st.session_state.config.pdf_path,
            st.session_state.config.page,
            st.session_state.config.zoom
        )
        st.session_state.current_image = processor.render_to_array()
    
    def extract_template(self):
        """Extract template from ROI"""
        if st.session_state.config.roi and st.session_state.current_image is not None:
            extractor = TemplateExtractor()
            st.session_state.current_template = extractor.extract_template(
                st.session_state.current_image,
                st.session_state.config.roi
            )
            st.session_state.template_extracted = True
    
    def run_detection(self):
        """Run symbol detection"""
        from src.detection.detection_pipeline import DetectionPipeline
        
        pipeline = DetectionPipeline(st.session_state.config)
        results = pipeline.run(
            st.session_state.current_image,
            st.session_state.current_template
        )
        st.session_state.detection_results = results
    
    def render_detection_preview(self):
        """Render detection preview with boxes"""
        from src.ui.visualization import draw_detections
        
        if st.session_state.detection_results and st.session_state.current_image is not None:
            annotated = draw_detections(
                st.session_state.current_image.copy(),
                st.session_state.detection_results.detections
            )
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )
    
    def render_results_table(self):
        """Render detailed results table"""
        import pandas as pd
        
        data = []
        for i, det in enumerate(st.session_state.detection_results.detections):
            data.append({
                'ID': i,
                'X1': det.x1,
                'Y1': det.y1,
                'X2': det.x2,
                'Y2': det.y2,
                'Confidence': f"{det.confidence:.3f}",
                'Scale': f"{det.scale:.2f}",
                'Angle': f"{det.angle:.1f}°"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    def save_configuration(self):
        """Save current configuration"""
        filepath = st.text_input("Configuration file path:", "config.json")
        if st.button("Save"):
            self.config_manager.config = st.session_state.config
            self.config_manager.save_to_file(filepath)
            st.success(f"✅ Configuration saved to {filepath}")
    
    def load_configuration(self):
        """Load configuration from file"""
        filepath = st.text_input("Configuration file path:", "config.json")
        if st.button("Load") and os.path.exists(filepath):
            self.config_manager.load_from_file(filepath)
            st.session_state.config = self.config_manager.config
            st.success(f"✅ Configuration loaded from {filepath}")
            st.experimental_rerun()

def main():
    """Main entry point"""
    app = SymbolDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
```

## Deliverables
1. Complete Streamlit application with all features
2. Support for all argparse parameters in UI
3. Interactive ROI selection
4. Real-time detection preview
5. Multiple export formats
6. Configuration save/load functionality

## Success Criteria
- All original argparse parameters accessible via UI
- Smooth workflow from PDF upload to results export
- Responsive and intuitive interface
- Proper error handling and user feedback
- Configuration persistence across sessions
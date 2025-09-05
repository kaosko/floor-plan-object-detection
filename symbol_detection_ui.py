#!/usr/bin/env python3
"""
Analytical Symbol Detection - Streamlit UI (Fixed Version)
Simplified interface with working PDF visualization and ROI selection
"""

import streamlit as st
import os
import tempfile
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import time
import subprocess
import sys
import fitz  # PyMuPDF for PDF processing
from streamlit_drawable_canvas import st_canvas

# Set page config
st.set_page_config(
    page_title="Analytical Symbol Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stButton > button {
    width: 100%;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    if 'roi_selected' not in st.session_state:
        st.session_state.roi_selected = False
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'roi_zoom_level' not in st.session_state:
        st.session_state.roi_zoom_level = 1.0
    if 'config' not in st.session_state:
        st.session_state.config = {
            'pdf_path': '',
            'page': 0,
            'zoom': 6.0,
            'threshold': 0.65,
            'scales': [0.95, 1.0, 1.05],
            'angles': [0.0],
            'method': 'CCOEFF_NORMED',
            'class_name': 'hvac',
            'use_edges': True,
            'roi': None,
            'coarse_scale': 0.5,
            'topk': 300,
            'refine_pad': 0.5,
            'output_dir': 'symbol_detection_output'
        }

def render_pdf_to_image(pdf_path: str, page_num: int, zoom: float):
    """Convert PDF page to OpenCV image"""
    try:
        pdf_doc = fitz.open(pdf_path)
        
        if page_num >= len(pdf_doc):
            pdf_doc.close()
            return False, f"Page {page_num} doesn't exist. PDF has {len(pdf_doc)} pages."
        
        # Get the page
        page = pdf_doc[page_num]
        
        # Create transformation matrix for zoom
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.tobytes("ppm")
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        pdf_doc.close()
        
        if img is None:
            return False, "Failed to decode image from PDF"
        
        return True, img
        
    except Exception as e:
        return False, str(e)

def render_interactive_roi_selection():
    """Interactive ROI selection using drawable canvas - Fixed Version"""
    st.subheader("🖱️ Interactive ROI Selection")
    
    if st.session_state.current_image is None:
        st.warning("⚠️ No rendered PDF image available for ROI selection")
        
        # Offer to render the PDF automatically
        if st.session_state.pdf_loaded:
            col1, col2 = st.columns(2)
            with col1:
                st.info("📄 PDF loaded but not yet rendered")
            with col2:
                if st.button("🖼️ Render PDF Now", type="primary"):
                    with st.spinner("Rendering PDF page..."):
                        success, image_or_error = render_pdf_to_image(
                            st.session_state.config['pdf_path'],
                            st.session_state.config['page'],
                            st.session_state.config['zoom']
                        )
                        
                        if success:
                            st.session_state.current_image = image_or_error
                            st.success("✅ PDF page rendered successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error rendering PDF: {image_or_error}")
        else:
            st.error("📄 Please upload a PDF file first using the sidebar")
        return
    
    # Add zoom controls
    st.markdown("### 🔍 Zoom Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍➕ Zoom In"):
            st.session_state.roi_zoom_level = min(st.session_state.roi_zoom_level * 1.5, 3.0)
            st.rerun()
    
    with col2:
        if st.button("🔍➖ Zoom Out"):
            st.session_state.roi_zoom_level = max(st.session_state.roi_zoom_level / 1.5, 0.5)
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Zoom"):
            st.session_state.roi_zoom_level = 1.0
            st.rerun()
    
    with col4:
        st.write(f"Zoom: {st.session_state.roi_zoom_level:.1f}x")
    
    # Zoom slider for precise control
    zoom_slider = st.slider(
        "Fine Zoom Control", 
        min_value=0.5, 
        max_value=3.0, 
        value=st.session_state.roi_zoom_level, 
        step=0.1,
        key="roi_zoom_slider"
    )
    
    if abs(zoom_slider - st.session_state.roi_zoom_level) > 0.01:
        st.session_state.roi_zoom_level = zoom_slider
        st.rerun()
    
    # Convert image to RGB for display
    display_image = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_image)
    
    # Calculate display size with zoom
    h, w = display_image.shape[:2]
    max_canvas_width = 800
    max_canvas_height = 600
    
    # Apply zoom to the base scale calculation
    zoom_level = st.session_state.roi_zoom_level
    
    # Calculate base display size
    if w > max_canvas_width or h > max_canvas_height:
        base_scale_w = max_canvas_width / w
        base_scale_h = max_canvas_height / h
        base_scale = min(base_scale_w, base_scale_h)
    else:
        base_scale = 1.0
    
    # Apply zoom to the base scale
    effective_scale = base_scale * zoom_level
    
    # Calculate final display dimensions
    display_width = int(w * effective_scale)
    display_height = int(h * effective_scale)
    
    # Limit canvas size to max dimensions
    canvas_width = min(display_width, max_canvas_width)
    canvas_height = min(display_height, max_canvas_height)
    
    # Resize image for display
    img_pil_resized = img_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
    
    # If image is larger than canvas, crop to center
    if display_width > max_canvas_width or display_height > max_canvas_height:
        # Calculate crop area to center the image
        crop_x = max(0, (display_width - canvas_width) // 2)
        crop_y = max(0, (display_height - canvas_height) // 2)
        img_pil_resized = img_pil_resized.crop((
            crop_x, 
            crop_y, 
            crop_x + canvas_width, 
            crop_y + canvas_height
        ))
        # Store crop offset for coordinate conversion
        crop_offset = (crop_x, crop_y)
    else:
        crop_offset = (0, 0)
    
    # Drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",  # Semi-transparent red fill
        stroke_width=2,
        stroke_color="#FF0000",  # Red stroke
        background_image=img_pil_resized,
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="rect",  # Rectangle drawing mode
        point_display_radius=0,
        key="roi_canvas_symbol_zoom",
    )
    
    # Process canvas data
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if objects:
            # Get the last drawn rectangle
            last_rect = objects[-1]
            
            if last_rect["type"] == "rect":
                # Extract rectangle coordinates (in canvas coordinates)
                canvas_x = int(last_rect["left"])
                canvas_y = int(last_rect["top"]) 
                canvas_width_rect = int(last_rect["width"])
                canvas_height_rect = int(last_rect["height"])
                
                # Convert from canvas coordinates to original image coordinates
                # Step 1: Add crop offset to get coordinates in the full scaled image
                scaled_x = canvas_x + crop_offset[0]
                scaled_y = canvas_y + crop_offset[1]
                
                # Step 2: Convert from scaled coordinates to original image coordinates
                roi_x = int(scaled_x / effective_scale)
                roi_y = int(scaled_y / effective_scale)
                roi_width = int(canvas_width_rect / effective_scale)
                roi_height = int(canvas_height_rect / effective_scale)
                
                # Ensure coordinates are within image bounds
                roi_x = max(0, min(roi_x, w - 1))
                roi_y = max(0, min(roi_y, h - 1))
                roi_width = max(10, min(roi_width, w - roi_x))
                roi_height = max(10, min(roi_height, h - roi_y))
                
                # Show selected ROI info
                st.info(f"Selected ROI: ({roi_x}, {roi_y}) - {roi_width}×{roi_height} px")
                
                # Show ROI crop preview
                roi_crop = st.session_state.current_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                if roi_crop.size > 0:
                    roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(roi_rgb, caption="Selected ROI Template", width=200)
                    
                    with col2:
                        st.write("**Template Quality Analysis:**")
                        
                        # Basic quality metrics
                        gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate some basic metrics
                        mean_intensity = np.mean(gray_roi)
                        std_intensity = np.std(gray_roi)
                        
                        # Edge density
                        edges = cv2.Canny(gray_roi, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        
                        st.write(f"• Size: {roi_width}×{roi_height} pixels")
                        st.write(f"• Mean intensity: {mean_intensity:.1f}")
                        st.write(f"• Intensity variation: {std_intensity:.1f}")
                        st.write(f"• Edge density: {edge_density:.3f}")
                        
                        # Quality assessment
                        quality_messages = []
                        
                        if roi_width >= 20 and roi_height >= 20:
                            quality_messages.append("✅ Good size")
                        else:
                            quality_messages.append("⚠️ Template might be too small")
                        
                        if std_intensity > 20:
                            quality_messages.append("✅ Good contrast")
                        else:
                            quality_messages.append("⚠️ Low contrast")
                        
                        if edge_density > 0.1:
                            quality_messages.append("✅ Rich in edges")
                        else:
                            quality_messages.append("⚠️ Few edges detected")
                        
                        for msg in quality_messages:
                            st.write(f"  {msg}")
                
                # Apply ROI button
                if st.button("🎯 Use This ROI for Detection", type="primary"):
                    st.session_state.config['roi'] = (roi_x, roi_y, roi_width, roi_height)
                    st.session_state.roi_selected = True
                    st.success(f"✅ ROI applied: ({roi_x}, {roi_y}, {roi_width}, {roi_height})")
                    st.rerun()
        else:
            st.info("👆 Draw a rectangle around your target symbol to select ROI")
    
    # Show current zoom level
    st.info(f"🔍 Current Zoom: {st.session_state.roi_zoom_level:.1f}x | Image Size: {w}×{h} px")
    
    # Instructions
    st.markdown("""
    ### 📖 Instructions
    
    **Zoom Controls:**
    - 🔍➕ Use Zoom In/Out buttons for quick zoom changes
    - 🎚️ Use the slider for precise zoom control (0.5x - 3.0x)
    - 🔄 Reset Zoom to return to original view
    
    **ROI Selection:**
    - 🖱️ Click and drag to draw a rectangle around a template symbol
    - 🎯 Select a clear, representative example of your target symbol
    - ✨ The red rectangle shows your selection
    - 🔄 Draw a new rectangle to replace the previous selection
    - 📏 Optimal template size: 50x50 to 200x200 pixels
    - 🔍 Use zoom to see small details clearly for precise selection
    """)

def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.header("Configuration")
    
    # File upload section
    st.sidebar.subheader("📄 PDF Input")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Select a floor plan PDF to analyze"
    )
    
    if uploaded_file:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.config['pdf_path'] = tmp_file.name
            st.session_state.pdf_loaded = True
            st.sidebar.success(f"✅ PDF loaded: {uploaded_file.name}")
    
    if st.session_state.pdf_loaded:
        # PDF settings
        st.sidebar.subheader("⚙️ PDF Settings")
        
        st.session_state.config['page'] = st.sidebar.number_input(
            "Page Number",
            min_value=0,
            max_value=100,
            value=st.session_state.config['page'],
            help="Select page (0-based index)"
        )
        
        st.session_state.config['zoom'] = st.sidebar.slider(
            "Zoom Factor",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.config['zoom'],
            step=0.5,
            help="Higher zoom = better quality but larger file"
        )
        
        # Detection parameters
        st.sidebar.subheader("🎯 Detection Parameters")
        
        st.session_state.config['threshold'] = st.sidebar.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.config['threshold'],
            step=0.05,
            help="Lower = more detections (may include false positives)"
        )
        
        # Symbol class selection
        st.sidebar.subheader("🏷️ Symbol Class")
        st.session_state.config['class_name'] = st.sidebar.selectbox(
            "Symbol Type",
            options=['hvac', 'door', 'window', 'electrical', 'plumbing', 'custom'],
            help="Select the type of symbol to detect"
        )

def run_analytical_detection():
    """Run the analytical symbol detection using the existing script"""
    try:
        # Prepare arguments for the analytical detection script
        args = [
            'python', 'analytical_symbol_detection.py',
            '--pdf', st.session_state.config['pdf_path'],
            '--page', str(st.session_state.config['page']),
            '--zoom', str(st.session_state.config['zoom']),
            '--threshold', str(st.session_state.config['threshold']),
            '--class-name', st.session_state.config['class_name'],
            '--outdir', st.session_state.config['output_dir']
        ]
        
        if st.session_state.config['roi']:
            args.extend(['--roi', ','.join(map(str, st.session_state.config['roi']))])
        
        # Run the detection
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Detection timed out after 5 minutes"
    except Exception as e:
        return False, str(e)

def render_main_content():
    """Render main content area"""
    # Create tabs for different stages
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 PDF Processing",
        "✂️ ROI Selection", 
        "🔍 Detection",
        "📊 Results"
    ])
    
    with tab1:
        render_pdf_processing_tab()
    
    with tab2:
        render_roi_selection_tab()
    
    with tab3:
        render_detection_tab()
    
    with tab4:
        render_results_tab()

def render_pdf_processing_tab():
    """Render PDF processing tab"""
    st.header("📄 PDF Processing")
    
    if not st.session_state.pdf_loaded:
        st.info("Please upload a PDF file using the sidebar.")
        return
    
    # PDF info and rendering
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ PDF loaded successfully")
        
        # Get PDF info
        try:
            pdf_doc = fitz.open(st.session_state.config['pdf_path'])
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            st.write(f"**Total Pages:** {total_pages}")
            st.write(f"**Current Page:** {st.session_state.config['page']}")
            st.write(f"**Zoom Factor:** {st.session_state.config['zoom']}")
            
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return
    
    with col2:
        if st.button("🖼️ Render PDF Page", type="primary"):
            with st.spinner("Rendering PDF page..."):
                success, image_or_error = render_pdf_to_image(
                    st.session_state.config['pdf_path'],
                    st.session_state.config['page'],
                    st.session_state.config['zoom']
                )
                
                if success:
                    st.session_state.current_image = image_or_error
                    st.success("✅ PDF page rendered successfully!")
                else:
                    st.error(f"❌ Error rendering PDF: {image_or_error}")
    
    # Display rendered image
    if st.session_state.current_image is not None:
        st.subheader("📷 Rendered Page")
        
        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
        
        # Show image info
        h, w = st.session_state.current_image.shape[:2]
        st.info(f"Image Size: {w} × {h} pixels")
        
        # Display the image
        st.image(display_image, caption=f"Page {st.session_state.config['page']}", use_column_width=True)

def render_roi_selection_tab():
    """Render ROI selection tab"""
    st.header("✂️ ROI Selection")
    
    if not st.session_state.pdf_loaded:
        st.warning("⚠️ Please upload and process a PDF first")
        return
    
    if st.session_state.current_image is None:
        st.warning("⚠️ Please render the PDF page first in the PDF Processing tab")
        return
    
    # Show current ROI if selected
    if st.session_state.roi_selected:
        roi = st.session_state.config['roi']
        st.info(f"Current ROI: ({roi[0]}, {roi[1]}) - {roi[2]}×{roi[3]} px")
    
    st.markdown("""
    **Instructions:**
    1. Draw a rectangle around a clear, representative example of your target symbol
    2. Select an area with minimal noise and overlapping elements
    3. The template should contain 1-2 complete symbol instances
    4. Avoid selecting areas with text or other symbols
    """)
    
    # Interactive ROI selection using drawable canvas
    render_interactive_roi_selection()
    
    st.markdown("---")
    
    # Manual ROI entry as fallback
    st.subheader("📝 Manual ROI Entry (Alternative)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x = st.number_input("X", min_value=0, value=0, key="roi_x")
    with col2:
        y = st.number_input("Y", min_value=0, value=0, key="roi_y")
    with col3:
        w = st.number_input("Width", min_value=1, value=100, key="roi_w")
    with col4:
        h = st.number_input("Height", min_value=1, value=100, key="roi_h")
    
    if st.button("Apply Manual ROI", type="secondary"):
        st.session_state.config['roi'] = (x, y, w, h)
        st.session_state.roi_selected = True
        st.success(f"✅ Manual ROI set: ({x}, {y}, {w}, {h})")

def render_detection_tab():
    """Render detection tab"""
    st.header("🔍 Symbol Detection")
    
    if not st.session_state.roi_selected:
        st.warning("⚠️ Please select ROI first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run Detection", type="primary", use_container_width=True):
            with st.spinner("Running symbol detection..."):
                success, message = run_analytical_detection()
                
                if success:
                    st.success("✅ Detection completed successfully!")
                    
                    # Show output messages
                    with st.expander("Detection Log"):
                        st.text(message)
                else:
                    st.error(f"❌ Detection failed: {message}")
    
    with col2:
        if st.session_state.roi_selected:
            roi = st.session_state.config['roi']
            st.info(f"ROI: {roi[0]},{roi[1]} - {roi[2]}x{roi[3]}")

def render_results_tab():
    """Render results tab"""
    st.header("📊 Detection Results")
    
    st.info("Results will be displayed here after running detection.")
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Output Folder", use_container_width=True):
            output_dir = st.session_state.config['output_dir']
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                st.write("**Output files:**")
                for file in files:
                    st.write(f"- {file}")
            else:
                st.error("Output directory not found")

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # App title and description
    st.title("🔍 Analytical Symbol Detection System")
    st.markdown("Modern interface for floor plan symbol detection using template matching")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
    
    # Main content
    if st.session_state.pdf_loaded:
        render_main_content()
    else:
        st.info("👋 Welcome! Please upload a PDF file using the sidebar to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | "
        "Powered by OpenCV template matching"
    )

if __name__ == "__main__":
    main()
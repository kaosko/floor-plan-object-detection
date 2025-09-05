#!/usr/bin/env python3
"""
UI Components for Analytical Symbol Detection System
Reusable Streamlit UI components for the detection interface
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.detection.data_classes import DetectionResults, Detection


def sidebar_config(config_manager) -> None:
    """
    Render configuration sidebar
    
    Args:
        config_manager: Configuration manager instance
    """
    st.sidebar.header("Configuration")
    
    # File upload section
    st.sidebar.subheader("📄 PDF Input")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Select a floor plan PDF to analyze"
    )
    
    if uploaded_file:
        # Process uploaded file
        st.session_state.uploaded_pdf = uploaded_file
        st.session_state.pdf_loaded = True
    
    if st.session_state.get('pdf_loaded', False):
        # PDF settings
        st.sidebar.subheader("⚙️ PDF Settings")
        
        page = st.sidebar.number_input(
            "Page Number",
            min_value=0,
            max_value=10,  # Will be updated dynamically
            value=st.session_state.config.page,
            help="Select page to analyze"
        )
        
        zoom = st.sidebar.slider(
            "Zoom Factor",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.config.zoom,
            step=0.5,
            help="Higher zoom = better quality but larger file"
        )
        
        # Update config
        st.session_state.config.page = page
        st.session_state.config.zoom = zoom
    
    # Detection parameters
    st.sidebar.subheader("🎯 Detection Parameters")
    
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.config.threshold,
        step=0.05,
        help="Lower = more detections (may include false positives)"
    )
    
    # Advanced settings in expander
    with st.sidebar.expander("🔧 Advanced Settings"):
        method = st.selectbox(
            "Matching Method",
            options=['CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'],
            index=0,
            help="Template matching algorithm"
        )
        
        use_edges = st.checkbox(
            "Use Edge Detection",
            value=st.session_state.config.use_edges,
            help="Match on edges instead of grayscale"
        )
        
        coarse_scale = st.slider(
            "Coarse Search Scale",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.config.coarse_scale,
            step=0.1,
            help="Downscale factor for initial search"
        )
        
        topk = st.number_input(
            "Top-K Candidates",
            min_value=10,
            max_value=1000,
            value=st.session_state.config.topk,
            step=10,
            help="Number of coarse candidates to refine"
        )
    
    # Update config
    st.session_state.config.threshold = threshold
    st.session_state.config.method = method
    st.session_state.config.use_edges = use_edges
    st.session_state.config.coarse_scale = coarse_scale
    st.session_state.config.topk = topk


def roi_selection_component(image: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    ROI selection component
    
    Args:
        image: Current image for ROI selection
        
    Returns:
        Selected ROI as (x, y, width, height) or None
    """
    st.subheader("ROI Selection")
    
    if image is None:
        st.warning("⚠️ Please render a PDF page first")
        return None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✂️ Select ROI Interactively", use_container_width=True):
            st.info("ROI selection window opened. Please select region in the popup window.")
            # In a real implementation, this would trigger the interactive ROI selection
            return None
    
    with col2:
        if st.button("📁 Load Saved ROI", use_container_width=True):
            # File uploader for ROI
            roi_file = st.file_uploader("Select ROI JSON file", type=['json'])
            if roi_file:
                import json
                roi_data = json.load(roi_file)
                return tuple(roi_data.get('roi', [0, 0, 100, 100]))
    
    with col3:
        manual_roi = st.checkbox("Enter ROI manually")
    
    if manual_roi:
        st.write("**Manual ROI Entry**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.number_input("X", min_value=0, value=0, key="roi_x")
        with col2:
            y = st.number_input("Y", min_value=0, value=0, key="roi_y")
        with col3:
            w = st.number_input("Width", min_value=1, value=100, key="roi_w")
        with col4:
            h = st.number_input("Height", min_value=1, value=100, key="roi_h")
        
        if st.button("Apply Manual ROI", type="primary"):
            return (x, y, w, h)
    
    return None


def detection_results_component(results: Optional[DetectionResults] = None) -> None:
    """
    Display detection results with visualizations
    
    Args:
        results: Detection results to display
    """
    st.subheader("Detection Results")
    
    if results is None:
        st.info("No detection results available. Run detection first.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", len(results.detections))
    
    with col2:
        avg_conf = results.get_average_confidence()
        st.metric("Avg Confidence", f"{avg_conf:.3f}")
    
    with col3:
        st.metric("Processing Time", f"{results.processing_time:.2f}s")
    
    with col4:
        conf_min, conf_max = results.get_confidence_range()
        st.metric("Confidence Range", f"{conf_min:.2f} - {conf_max:.2f}")
    
    # Detailed results table
    if st.checkbox("Show Detailed Results"):
        render_results_table(results)
    
    # Confidence distribution
    if len(results.detections) > 0:
        render_confidence_distribution(results)


def render_results_table(results: DetectionResults) -> None:
    """
    Render detailed results table
    
    Args:
        results: Detection results
    """
    st.write("**Detection Details**")
    
    data = []
    for i, det in enumerate(results.detections):
        data.append({
            'ID': i + 1,
            'X1': det.x1,
            'Y1': det.y1,
            'X2': det.x2,
            'Y2': det.y2,
            'Width': det.width,
            'Height': det.height,
            'Area': det.area,
            'Confidence': f"{det.confidence:.3f}",
            'Scale': f"{det.scale:.2f}",
            'Angle': f"{det.angle:.1f}°",
            'Class': det.class_name
        })
    
    df = pd.DataFrame(data)
    
    # Allow filtering by confidence
    min_conf = st.slider(
        "Filter by minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="conf_filter"
    )
    
    filtered_df = df[df['Confidence'].astype(float) >= min_conf]
    st.dataframe(filtered_df, use_container_width=True, height=300)
    
    # Download button for CSV
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📊 Download as CSV",
        data=csv_data,
        file_name="detection_results.csv",
        mime="text/csv"
    )


def render_confidence_distribution(results: DetectionResults) -> None:
    """
    Render confidence distribution visualization
    
    Args:
        results: Detection results
    """
    st.write("**Confidence Distribution**")
    
    confidences = [det.confidence for det in results.detections]
    
    # Create histogram
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Detection Confidence Distribution",
        labels={'x': 'Confidence', 'y': 'Count'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{np.mean(confidences):.3f}")
    with col2:
        st.metric("Std Dev", f"{np.std(confidences):.3f}")
    with col3:
        st.metric("Median", f"{np.median(confidences):.3f}")


def visualization_component(image: np.ndarray, 
                          detections: List[Detection],
                          show_confidence: bool = True,
                          color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Visualization component for detection results
    
    Args:
        image: Base image
        detections: List of detections to visualize
        show_confidence: Whether to show confidence scores
        color: Color for bounding boxes (BGR)
    """
    st.subheader("Detection Visualization")
    
    if len(detections) == 0:
        st.info("No detections to visualize")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        return
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    with col2:
        show_conf = st.checkbox("Show Confidence Scores", value=show_confidence)
    with col3:
        show_ids = st.checkbox("Show Detection IDs", value=False)
    
    # Color selection
    color_option = st.selectbox(
        "Box Color",
        options=['Green', 'Red', 'Blue', 'Yellow', 'Cyan', 'Magenta'],
        index=0
    )
    
    color_map = {
        'Green': (0, 255, 0),
        'Red': (0, 0, 255),
        'Blue': (255, 0, 0),
        'Yellow': (0, 255, 255),
        'Cyan': (255, 255, 0),
        'Magenta': (255, 0, 255)
    }
    
    box_color = color_map[color_option]
    
    # Create annotated image
    annotated_image = image.copy()
    
    for i, detection in enumerate(detections):
        if show_boxes:
            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                box_color,
                2
            )
        
        # Add text annotations
        text_y = detection.y1 - 5 if detection.y1 > 20 else detection.y1 + 20
        
        if show_conf:
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(
                annotated_image,
                conf_text,
                (detection.x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1
            )
        
        if show_ids:
            id_text = f"#{i+1}"
            text_x = detection.x2 - 30
            cv2.putText(
                annotated_image,
                id_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1
            )
    
    # Convert to RGB for display
    display_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(display_image, use_column_width=True)
    
    # Detection statistics overlay
    if st.checkbox("Show Detection Statistics Overlay"):
        render_detection_statistics_overlay(detections)


def render_detection_statistics_overlay(detections: List[Detection]) -> None:
    """
    Render detection statistics as an overlay
    
    Args:
        detections: List of detections
    """
    st.write("**Detection Statistics**")
    
    if not detections:
        return
    
    # Create statistics
    areas = [det.area for det in detections]
    confidences = [det.confidence for det in detections]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Area Distribution', 'Confidence vs Area'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Area histogram
    fig.add_trace(
        go.Histogram(x=areas, name="Area", nbinsx=10),
        row=1, col=1
    )
    
    # Scatter plot: confidence vs area
    fig.add_trace(
        go.Scatter(
            x=areas, y=confidences,
            mode='markers',
            name="Detections",
            text=[f"Det {i+1}" for i in range(len(detections))],
            hovertemplate="Area: %{x}<br>Confidence: %{y}<br>%{text}"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=300, showlegend=False)
    fig.update_xaxes(title_text="Area (pixels²)", row=1, col=1)
    fig.update_xaxes(title_text="Area (pixels²)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def progress_indicator(stage: str, progress: float, message: str = "") -> None:
    """
    Show progress indicator for detection stages
    
    Args:
        stage: Current processing stage
        progress: Progress value (0.0 to 1.0)
        message: Optional message to display
    """
    st.write(f"**{stage}**")
    
    progress_bar = st.progress(progress)
    
    if message:
        st.write(message)
    
    return progress_bar


def export_options_component(results: Optional[DetectionResults] = None,
                           image: Optional[np.ndarray] = None) -> None:
    """
    Export options component
    
    Args:
        results: Detection results
        image: Original image
    """
    st.subheader("Export Options")
    
    if results is None:
        st.info("No results to export. Run detection first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Export YOLO Labels", use_container_width=True):
            export_yolo_labels(results)
    
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            export_csv_results(results)
    
    with col3:
        if st.button("🖼️ Save Annotated Image", use_container_width=True):
            if image is not None:
                save_annotated_image(image, results)
            else:
                st.error("No image available for annotation")


def export_yolo_labels(results: DetectionResults) -> None:
    """Export results in YOLO format"""
    # This would implement YOLO export
    st.info("YOLO export functionality would be implemented here")


def export_csv_results(results: DetectionResults) -> None:
    """Export results as CSV"""
    data = []
    for i, det in enumerate(results.detections):
        data.append(det.to_dict())
    
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        label="📊 Download CSV",
        data=csv_data,
        file_name="detection_results.csv",
        mime="text/csv"
    )
    st.success("CSV ready for download!")


def save_annotated_image(image: np.ndarray, results: DetectionResults) -> None:
    """Save annotated image"""
    # This would implement image saving
    st.info("Annotated image save functionality would be implemented here")


def error_display_component(error_message: str, suggestions: List[str] = None) -> None:
    """
    Display error messages with helpful suggestions
    
    Args:
        error_message: Error message to display
        suggestions: Optional list of suggestions
    """
    st.error(f"❌ Error: {error_message}")
    
    if suggestions:
        st.write("**Suggestions:**")
        for suggestion in suggestions:
            st.write(f"• {suggestion}")


def help_component() -> None:
    """Render help information"""
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use the Symbol Detection System
        
        1. **Upload PDF**: Use the sidebar to upload a floor plan PDF
        2. **Configure Settings**: Adjust detection parameters in the sidebar
        3. **Render Page**: Click "Render PDF Page" to process the PDF
        4. **Select ROI**: Choose a region containing your target symbol
        5. **Run Detection**: Click "Run Detection" to find symbols
        6. **View Results**: Analyze results in the Results tab
        7. **Export**: Save results in various formats
        
        ### Tips for Better Results
        
        - **Template Quality**: Select a clear, well-defined symbol region
        - **Detection Threshold**: Lower values find more symbols but may include false positives
        - **Use Edges**: Often works better for architectural drawings
        - **Multiple Scales**: Include scales around 1.0 (e.g., 0.9, 1.0, 1.1)
        - **Coarse Scale**: Use 0.3-0.7 for faster processing on large images
        
        ### Troubleshooting
        
        - **No Detections**: Try lowering the threshold or adjusting scales
        - **Too Many False Positives**: Increase threshold or improve template selection
        - **Slow Processing**: Reduce coarse scale or limit search angles
        - **Poor Results**: Check if "Use Edge Detection" helps for your drawings
        """)


def status_indicator(status: str, color: str = "blue") -> None:
    """
    Show status indicator
    
    Args:
        status: Status message
        color: Color for the indicator
    """
    color_map = {
        "blue": "🔵",
        "green": "🟢", 
        "red": "🔴",
        "yellow": "🟡",
        "orange": "🟠"
    }
    
    icon = color_map.get(color, "🔵")
    st.write(f"{icon} {status}")


def template_preview_component(template: Optional[np.ndarray] = None,
                             template_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Template preview component
    
    Args:
        template: Template image
        template_info: Template analysis information
    """
    st.subheader("Template Preview")
    
    if template is None:
        st.info("No template available. Please select ROI first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display template
        if len(template.shape) == 3:
            display_template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        else:
            display_template = template
        
        st.image(display_template, caption="Extracted Template", use_column_width=True)
    
    with col2:
        if template_info:
            st.write("**Template Analysis**")
            
            # Quality score
            quality = template_info.get('quality_score', 0.0)
            quality_color = "green" if quality > 0.7 else "yellow" if quality > 0.4 else "red"
            st.metric("Quality Score", f"{quality:.2f}", delta_color=quality_color)
            
            # Key metrics
            st.write(f"- **Size**: {template_info.get('shape', 'Unknown')}")
            st.write(f"- **Mean Intensity**: {template_info.get('mean', 0):.1f}")
            st.write(f"- **Edge Density**: {template_info.get('edge_density', 0):.3f}")
            st.write(f"- **Contours**: {template_info.get('num_contours', 0)}")
            
            # Quality assessment
            if quality > 0.7:
                st.success("✅ Good template quality")
            elif quality > 0.4:
                st.warning("⚠️ Moderate template quality")
            else:
                st.error("❌ Poor template quality - consider re-selecting ROI")
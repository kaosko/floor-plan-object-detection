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
import base64
import streamlit.components.v1 as components

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
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
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

def image_to_base64(cv_image):
    """Convert OpenCV image to base64 string."""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    # Convert to base64
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def handle_template_update(component_value):
    """Handle updates from the template annotation interface."""
    if isinstance(component_value, dict):
        if component_value.get('action') == 'update_templates':
            st.session_state.annotations = component_value.get('templates', [])
        elif component_value.get('action') == 'use_template':
            template = component_value.get('template')
            if template and template.get('type') == 'rectangle':
                # Set ROI from template
                st.session_state.config['roi'] = (
                    int(template['x']), int(template['y']),
                    int(template['width']), int(template['height'])
                )
                st.session_state.roi_selected = True
                st.success(f"✅ Template T{template['id']} applied as ROI!")
                st.rerun()

def render_interactive_roi_selection():
    """Native HTML/JS annotation interface for symbol templates."""
    st.subheader("🖱️ Native Template Selection Interface")
    
    if st.session_state.current_image is None:
        st.warning("⚠️ No rendered PDF image available for template selection")
        
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
    
    # Convert image to base64 for embedding
    image_base64 = image_to_base64(st.session_state.current_image)
    
    # Get current annotations from session state
    annotations_json = json.dumps(st.session_state.annotations)
    
    html_content = f"""
    <div id="symbol-annotation-container" style="width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
        <div style="display: grid; grid-template-rows: 50px 1fr 120px; height: 100%;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; display: flex; align-items: center; justify-content: space-between; padding: 0 15px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px;">🔍</span>
                    <span style="font-size: 14px; font-weight: 600;">Symbol Template Selection</span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button id="saveTemplateBtn" style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">Save Template</button>
                    <button id="clearBtn" style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">Clear</button>
                </div>
            </div>
            
            <!-- Main Content -->
            <div style="display: grid; grid-template-columns: 150px 1fr; height: 100%;">
                <!-- Sidebar -->
                <div style="background: #f8f9fa; border-right: 1px solid #ddd; padding: 10px; overflow-y: auto;">
                    <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 8px;">TOOLS</div>
                    <button class="tool-btn active" data-tool="select" style="display: block; width: 100%; padding: 6px; margin-bottom: 4px; background: #007bff; color: white; border: 1px solid #0056b3; border-radius: 4px; cursor: pointer; font-size: 11px;">↖ Select</button>
                    <button class="tool-btn" data-tool="rectangle" style="display: block; width: 100%; padding: 6px; margin-bottom: 4px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; cursor: pointer; font-size: 11px;">⬜ Rectangle</button>
                    <div style="margin-top: 15px;">
                        <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 8px;">VIEW</div>
                        <button id="zoomInBtn" style="display: block; width: 100%; padding: 6px; margin-bottom: 4px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; cursor: pointer; font-size: 11px;">🔍 Zoom In</button>
                        <button id="zoomOutBtn" style="display: block; width: 100%; padding: 6px; margin-bottom: 4px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; cursor: pointer; font-size: 11px;">🔍 Zoom Out</button>
                        <button id="fitBtn" style="display: block; width: 100%; padding: 6px; margin-bottom: 4px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; cursor: pointer; font-size: 11px;">📐 Fit</button>
                    </div>
                </div>
                
                <!-- Canvas -->
                <div style="position: relative; overflow: hidden; background: #fff;">
                    <canvas id="symbolCanvas" style="position: absolute; top: 0; left: 0; cursor: crosshair;"></canvas>
                    <div id="symbolStatusBar" style="position: absolute; bottom: 5px; left: 5px; background: rgba(0,0,0,0.7); color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
                        Ready | Tool: Select | Templates: 0
                    </div>
                </div>
            </div>
            
            <!-- Bottom Panel -->
            <div style="background: #fff; border-top: 1px solid #ddd; padding: 8px; overflow-y: auto;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 12px; font-weight: 600; color: #333;">TEMPLATE MANAGER</span>
                    <div>
                        <button id="useTemplateBtn" style="background: #28a745; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 10px; margin-right: 4px;">Use Selected</button>
                        <button id="deleteTemplateBtn" style="background: #dc3545; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 10px;">Delete</button>
                    </div>
                </div>
                <div id="templateList" style="font-size: 11px;">
                    <div style="color: #666;">No templates created yet. Draw rectangles around target symbols to create templates.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        // Symbol template annotation interface
        let canvas, ctx;
        let currentTool = 'select';
        let isDrawing = false;
        let zoom = 1.0;
        let panX = 0, panY = 0;
        let templates = {annotations_json};
        let selectedTemplate = null;
        let nextId = 1;
        let backgroundImage = null;
        let startX, startY;
        
        function init() {{
            canvas = document.getElementById('symbolCanvas');
            if (!canvas) return;
            
            ctx = canvas.getContext('2d');
            
            // Load background image
            const img = new Image();
            img.onload = function() {{
                backgroundImage = img;
                resizeCanvas();
                redraw();
            }};
            img.src = 'data:image/jpeg;base64,{image_base64}';
            
            // Bind events
            bindEvents();
            updateStatus();
            updateTemplateList();
        }}
        
        function resizeCanvas() {{
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            
            if (backgroundImage) {{
                // Fit image to canvas
                const scaleX = canvas.width / backgroundImage.width;
                const scaleY = canvas.height / backgroundImage.height;
                zoom = Math.min(scaleX, scaleY) * 0.8;
                panX = (canvas.width - backgroundImage.width * zoom) / 2;
                panY = (canvas.height - backgroundImage.height * zoom) / 2;
            }}
            
            redraw();
        }}
        
        function bindEvents() {{
            // Canvas events
            canvas.addEventListener('mousedown', handleMouseDown);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseup', handleMouseUp);
            canvas.addEventListener('wheel', handleWheel);
            
            // Tool buttons
            document.querySelectorAll('.tool-btn').forEach(btn => {{
                btn.addEventListener('click', () => selectTool(btn.dataset.tool));
            }});
            
            // Action buttons
            document.getElementById('zoomInBtn').addEventListener('click', () => zoomBy(1.2));
            document.getElementById('zoomOutBtn').addEventListener('click', () => zoomBy(0.8));
            document.getElementById('fitBtn').addEventListener('click', fitToView);
            document.getElementById('saveTemplateBtn').addEventListener('click', saveTemplate);
            document.getElementById('clearBtn').addEventListener('click', clearTemplates);
            document.getElementById('useTemplateBtn').addEventListener('click', useSelectedTemplate);
            document.getElementById('deleteTemplateBtn').addEventListener('click', deleteSelectedTemplate);
            
            // Window resize
            window.addEventListener('resize', resizeCanvas);
        }}
        
        function selectTool(tool) {{
            currentTool = tool;
            
            // Update UI
            document.querySelectorAll('.tool-btn').forEach(btn => {{
                btn.style.background = btn.dataset.tool === tool ? '#007bff' : '#f8f9fa';
                btn.style.color = btn.dataset.tool === tool ? 'white' : 'black';
            }});
            
            canvas.style.cursor = tool === 'select' ? 'default' : 'crosshair';
            updateStatus();
        }}
        
        function handleMouseDown(e) {{
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            
            startX = x;
            startY = y;
            
            if (currentTool === 'rectangle') {{
                isDrawing = true;
            }} else if (currentTool === 'select') {{
                selectedTemplate = getTemplateAt(x, y);
                updateTemplateList();
                redraw();
            }}
        }}
        
        function handleMouseMove(e) {{
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            
            if (currentTool === 'rectangle') {{
                redraw();
                drawPreviewRect(startX, startY, x - startX, y - startY);
            }}
        }}
        
        function handleMouseUp(e) {{
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            
            if (currentTool === 'rectangle') {{
                const width = x - startX;
                const height = y - startY;
                
                if (Math.abs(width) > 10 && Math.abs(height) > 10) {{
                    createTemplate(
                        Math.min(startX, x),
                        Math.min(startY, y),
                        Math.abs(width),
                        Math.abs(height)
                    );
                }}
            }}
            
            isDrawing = false;
        }}
        
        function handleWheel(e) {{
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(5, zoom * zoomFactor));
            
            panX = x - (x - panX) * (newZoom / zoom);
            panY = y - (y - panY) * (newZoom / zoom);
            
            zoom = newZoom;
            redraw();
        }}
        
        function redraw() {{
            if (!ctx || !canvas) return;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);
            
            // Draw background image
            if (backgroundImage) {{
                ctx.drawImage(backgroundImage, 0, 0);
            }}
            
            // Draw templates
            templates.forEach(template => {{
                if (!template.visible) return;
                
                ctx.save();
                ctx.strokeStyle = template === selectedTemplate ? '#ff0000' : '#00ff00';
                ctx.lineWidth = 2;
                ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
                
                ctx.fillRect(template.x, template.y, template.width, template.height);
                ctx.strokeRect(template.x, template.y, template.width, template.height);
                
                // Draw ID
                ctx.fillStyle = '#000';
                ctx.font = '12px Arial';
                ctx.fillText(`T${{template.id}}`, template.x + 5, template.y + 15);
                
                ctx.restore();
            }});
            
            ctx.restore();
        }}
        
        function drawPreviewRect(x, y, w, h) {{
            ctx.save();
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x, y, w, h);
            ctx.restore();
        }}
        
        function createTemplate(x, y, width, height) {{
            const template = {{
                id: nextId++,
                type: 'rectangle',
                label: `Template ${{nextId - 1}}`,
                x: x,
                y: y,
                width: width,
                height: height,
                visible: true,
                color: '#00ff00'
            }};
            
            templates.push(template);
            redraw();
            updateTemplateList();
            updateStatus();
            sendToStreamlit();
        }}
        
        function getTemplateAt(x, y) {{
            for (let i = templates.length - 1; i >= 0; i--) {{
                const template = templates[i];
                if (!template.visible) continue;
                
                if (x >= template.x && x <= template.x + template.width &&
                    y >= template.y && y <= template.y + template.height) {{
                    return template;
                }}
            }}
            return null;
        }}
        
        function updateStatus() {{
            const statusBar = document.getElementById('symbolStatusBar');
            const toolName = currentTool.charAt(0).toUpperCase() + currentTool.slice(1);
            statusBar.textContent = `Ready | Tool: ${{toolName}} | Templates: ${{templates.length}}`;
        }}
        
        function updateTemplateList() {{
            const templateList = document.getElementById('templateList');
            
            if (templates.length === 0) {{
                templateList.innerHTML = '<div style="color: #666;">No templates created yet. Draw rectangles around target symbols to create templates.</div>';
                return;
            }}
            
            let html = '';
            templates.forEach(template => {{
                const isSelected = template === selectedTemplate;
                const bgColor = isSelected ? '#e3f2fd' : 'transparent';
                const quality = analyzeTemplateQuality(template);
                html += `
                    <div style="display: flex; align-items: center; padding: 4px; background: ${{bgColor}}; border-radius: 4px; margin-bottom: 2px; cursor: pointer;" onclick="selectTemplate(${{template.id}})">
                        <span style="min-width: 30px; font-weight: 500;">T${{template.id}}</span>
                        <span style="flex: 1; margin-left: 8px;">${{template.label}} (${{template.width}}×${{template.height}}px)</span>
                        <span style="font-size: 9px; color: ${{quality.color}}; margin-left: 4px;">${{quality.text}}</span>
                        <button onclick="event.stopPropagation(); toggleTemplateVisibility(${{template.id}})" style="background: none; border: none; cursor: pointer; font-size: 12px; margin-left: 4px;">${{template.visible ? '👁' : '🚫'}}</button>
                    </div>
                `;
            }});
            
            templateList.innerHTML = html;
        }}
        
        function analyzeTemplateQuality(template) {{
            const area = template.width * template.height;
            if (area < 400) return {{color: '#dc3545', text: 'Small'}};
            if (area > 40000) return {{color: '#ffc107', text: 'Large'}};
            return {{color: '#28a745', text: 'Good'}};
        }}
        
        function selectTemplate(id) {{
            selectedTemplate = templates.find(t => t.id === id) || null;
            redraw();
            updateTemplateList();
        }}
        
        function toggleTemplateVisibility(id) {{
            const template = templates.find(t => t.id === id);
            if (template) {{
                template.visible = !template.visible;
                redraw();
                updateTemplateList();
                sendToStreamlit();
            }}
        }}
        
        function useSelectedTemplate() {{
            if (selectedTemplate) {{
                const templateData = {{
                    action: 'use_template',
                    template: selectedTemplate
                }};
                
                window.parent.postMessage({{
                    type: 'streamlit:componentValue',
                    value: templateData
                }}, '*');
                
                // Visual feedback
                const btn = document.getElementById('useTemplateBtn');
                const originalText = btn.textContent;
                btn.textContent = 'Applied!';
                btn.style.background = '#007bff';
                setTimeout(() => {{
                    btn.textContent = originalText;
                    btn.style.background = '#28a745';
                }}, 1000);
            }}
        }}
        
        function deleteSelectedTemplate() {{
            if (selectedTemplate) {{
                const index = templates.indexOf(selectedTemplate);
                if (index > -1) {{
                    templates.splice(index, 1);
                    selectedTemplate = null;
                    redraw();
                    updateTemplateList();
                    updateStatus();
                    sendToStreamlit();
                }}
            }}
        }}
        
        function saveTemplate() {{
            sendToStreamlit();
            // Visual feedback
            const btn = document.getElementById('saveTemplateBtn');
            const originalText = btn.textContent;
            btn.textContent = 'Saved!';
            btn.style.background = '#28a745';
            setTimeout(() => {{
                btn.textContent = originalText;
                btn.style.background = 'rgba(255,255,255,0.2)';
            }}, 1000);
        }}
        
        function clearTemplates() {{
            if (confirm('Clear all templates?')) {{
                templates = [];
                selectedTemplate = null;
                redraw();
                updateTemplateList();
                updateStatus();
                sendToStreamlit();
            }}
        }}
        
        function zoomBy(factor) {{
            zoom *= factor;
            zoom = Math.max(0.1, Math.min(5, zoom));
            redraw();
        }}
        
        function fitToView() {{
            if (!backgroundImage) return;
            
            const scaleX = canvas.width / backgroundImage.width;
            const scaleY = canvas.height / backgroundImage.height;
            zoom = Math.min(scaleX, scaleY) * 0.9;
            
            panX = (canvas.width - backgroundImage.width * zoom) / 2;
            panY = (canvas.height - backgroundImage.height * zoom) / 2;
            
            redraw();
        }}
        
        function sendToStreamlit() {{
            const data = {{
                action: 'update_templates',
                templates: templates
            }};
            
            window.parent.postMessage({{
                type: 'streamlit:componentValue',
                value: data
            }}, '*');
        }}
        
        // Global functions for external access
        window.selectTemplate = selectTemplate;
        window.toggleTemplateVisibility = toggleTemplateVisibility;
        
        // Initialize
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            init();
        }}
    }})();
    </script>
    """
    
    # Render the HTML component
    component_value = components.html(html_content, height=600)
    
    # Handle component communication
    if component_value:
        handle_template_update(component_value)

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
    """Render template selection tab"""
    st.header("✂️ Symbol Template Selection")
    
    if not st.session_state.pdf_loaded:
        st.warning("⚠️ Please upload and process a PDF first")
        return
    
    if st.session_state.current_image is None:
        st.warning("⚠️ Please render the PDF page first in the PDF Processing tab")
        return
    
    # Show current ROI if selected
    if st.session_state.roi_selected:
        roi = st.session_state.config['roi']
        st.info(f"✅ Current Template ROI: ({roi[0]}, {roi[1]}) - {roi[2]}×{roi[3]} px")
    
    st.markdown("""
    **Instructions:**
    1. 🖱️ Select the rectangle tool from the sidebar in the interface below
    2. 🎯 Draw rectangles around clear examples of your target symbols
    3. 📏 Templates should be 50-200 pixels in size for best results
    4. ✅ Click "Use Selected" to apply a template as ROI for detection
    """)
    
    # Native annotation interface
    render_interactive_roi_selection()
    
    # Show current templates
    if st.session_state.annotations:
        st.subheader("📋 Created Templates")
        for template in st.session_state.annotations:
            if template.get('type') == 'rectangle':
                area = template['width'] * template['height']
                if area < 400:
                    quality = "⚠️ Small"
                elif area > 40000:
                    quality = "⚠️ Large" 
                else:
                    quality = "✅ Good"
                st.write(f"**T{template['id']}**: {template['label']} - "
                       f"({template['x']:.0f}, {template['y']:.0f}) {template['width']:.0f}×{template['height']:.0f}px - {quality}")
    
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
    st.title("🔍 Enhanced Symbol Detection System")
    st.markdown("Native annotation interface for floor plan symbol detection using template matching")
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
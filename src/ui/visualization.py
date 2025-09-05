#!/usr/bin/env python3
"""
Visualization utilities for Analytical Symbol Detection System
Functions for drawing detections, overlays, and creating visualizations
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

from src.detection.data_classes import Detection, DetectionResults


def draw_detections(image: np.ndarray, 
                   detections: List[Detection],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2,
                   show_confidence: bool = True,
                   show_class: bool = True,
                   font_scale: float = 0.6) -> np.ndarray:
    """
    Draw detection bounding boxes on image
    
    Args:
        image: Input image (BGR)
        detections: List of detections to draw
        color: Box color (BGR)
        thickness: Line thickness
        show_confidence: Show confidence scores
        show_class: Show class names
        font_scale: Font size for text
        
    Returns:
        Image with drawn detections
    """
    result = image.copy()
    
    for i, detection in enumerate(detections):
        # Draw bounding box
        cv2.rectangle(
            result,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            thickness
        )
        
        # Prepare text
        text_parts = []
        if show_class:
            text_parts.append(detection.class_name)
        if show_confidence:
            text_parts.append(f"{detection.confidence:.2f}")
        
        if text_parts:
            text = " ".join(text_parts)
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Position text above the box if possible
            text_x = detection.x1
            text_y = detection.y1 - 5 if detection.y1 > text_height + 5 else detection.y1 + text_height + 5
            
            # Draw text background
            cv2.rectangle(
                result,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                result,
                text,
                (text_x, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                1
            )
    
    return result


def draw_detection_with_id(image: np.ndarray,
                          detection: Detection,
                          detection_id: int,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
    """
    Draw single detection with ID number
    
    Args:
        image: Input image
        detection: Detection to draw
        detection_id: ID number to display
        color: Box color
        thickness: Line thickness
        
    Returns:
        Image with drawn detection
    """
    result = image.copy()
    
    # Draw bounding box
    cv2.rectangle(
        result,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        thickness
    )
    
    # Draw ID in top-left corner of box
    id_text = f"#{detection_id}"
    cv2.putText(
        result,
        id_text,
        (detection.x1 + 2, detection.y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )
    
    return result


def create_detection_heatmap(image: np.ndarray,
                           detections: List[Detection],
                           cell_size: int = 50) -> np.ndarray:
    """
    Create detection density heatmap
    
    Args:
        image: Base image
        detections: List of detections
        cell_size: Size of heatmap cells in pixels
        
    Returns:
        Heatmap overlay image
    """
    height, width = image.shape[:2]
    
    # Create grid
    grid_h = height // cell_size + 1
    grid_w = width // cell_size + 1
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    # Count detections in each cell
    for detection in detections:
        center_x = (detection.x1 + detection.x2) // 2
        center_y = (detection.y1 + detection.y2) // 2
        
        grid_x = min(center_x // cell_size, grid_w - 1)
        grid_y = min(center_y // cell_size, grid_h - 1)
        
        heatmap[grid_y, grid_x] += 1
    
    # Normalize and resize to original image size
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Blend with original image
    alpha = 0.6
    result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return result


def draw_roi_rectangle(image: np.ndarray,
                      roi: Tuple[int, int, int, int],
                      color: Tuple[int, int, int] = (255, 0, 0),
                      thickness: int = 3,
                      label: str = "ROI") -> np.ndarray:
    """
    Draw ROI rectangle on image
    
    Args:
        image: Input image
        roi: ROI as (x, y, width, height)
        color: Rectangle color
        thickness: Line thickness
        label: Text label for ROI
        
    Returns:
        Image with drawn ROI
    """
    result = image.copy()
    x, y, w, h = roi
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    cv2.putText(
        result,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    
    return result


def create_confidence_visualization(detections: List[Detection],
                                  image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create visualization showing confidence levels as colors
    
    Args:
        detections: List of detections
        image_shape: Shape of the base image (height, width)
        
    Returns:
        Confidence visualization image
    """
    height, width = image_shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    for detection in detections:
        # Map confidence to color (red = low, yellow = medium, green = high)
        conf = detection.confidence
        
        if conf < 0.5:
            # Red for low confidence
            color = (0, 0, int(255 * (conf / 0.5)))
        elif conf < 0.8:
            # Yellow for medium confidence
            ratio = (conf - 0.5) / 0.3
            color = (0, int(255 * ratio), 255)
        else:
            # Green for high confidence
            ratio = (conf - 0.8) / 0.2
            color = (0, 255, int(255 * (1 - ratio)))
        
        # Draw filled rectangle
        cv2.rectangle(
            result,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            -1  # Filled
        )
    
    return result


def create_scale_angle_visualization(detections: List[Detection],
                                   image: np.ndarray) -> np.ndarray:
    """
    Create visualization showing scale and angle information
    
    Args:
        detections: List of detections
        image: Base image
        
    Returns:
        Annotated image with scale/angle info
    """
    result = image.copy()
    
    for detection in detections:
        # Different colors for different scales
        if detection.scale < 0.9:
            color = (255, 0, 0)  # Blue for small
        elif detection.scale > 1.1:
            color = (0, 0, 255)  # Red for large
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Draw bounding box
        cv2.rectangle(
            result,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2
        )
        
        # Add scale and angle text
        info_text = f"S:{detection.scale:.2f} A:{detection.angle:.0f}°"
        cv2.putText(
            result,
            info_text,
            (detection.x1, detection.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
    
    return result


def create_detection_comparison(image1: np.ndarray,
                              image2: np.ndarray,
                              detections1: List[Detection],
                              detections2: List[Detection],
                              title1: str = "Image 1",
                              title2: str = "Image 2") -> np.ndarray:
    """
    Create side-by-side comparison of two detection results
    
    Args:
        image1: First image
        image2: Second image  
        detections1: Detections for first image
        detections2: Detections for second image
        title1: Title for first image
        title2: Title for second image
        
    Returns:
        Combined comparison image
    """
    # Draw detections on both images
    img1_with_det = draw_detections(image1, detections1, color=(0, 255, 0))
    img2_with_det = draw_detections(image2, detections2, color=(0, 255, 0))
    
    # Ensure images are same height
    h1, w1 = img1_with_det.shape[:2]
    h2, w2 = img2_with_det.shape[:2]
    
    target_height = max(h1, h2)
    
    if h1 != target_height:
        img1_with_det = cv2.resize(img1_with_det, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        img2_with_det = cv2.resize(img2_with_det, (int(w2 * target_height / h2), target_height))
    
    # Concatenate horizontally
    result = np.hstack([img1_with_det, img2_with_det])
    
    # Add titles
    cv2.putText(result, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, title2, (img1_with_det.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add detection counts
    count_text1 = f"Detections: {len(detections1)}"
    count_text2 = f"Detections: {len(detections2)}"
    
    cv2.putText(result, count_text1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, count_text2, (img1_with_det.shape[1] + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result


def create_template_matching_visualization(image: np.ndarray,
                                         template: np.ndarray,
                                         detections: List[Detection]) -> np.ndarray:
    """
    Create visualization showing template and matching results
    
    Args:
        image: Original image
        template: Template used for matching
        detections: Detection results
        
    Returns:
        Combined visualization
    """
    # Resize template for display
    template_display_size = 150
    th, tw = template.shape[:2]
    scale = min(template_display_size / tw, template_display_size / th)
    
    new_tw = int(tw * scale)
    new_th = int(th * scale)
    template_resized = cv2.resize(template, (new_tw, new_th))
    
    # Convert template to 3-channel if needed
    if len(template_resized.shape) == 2:
        template_resized = cv2.cvtColor(template_resized, cv2.COLOR_GRAY2BGR)
    
    # Draw detections on image
    result = draw_detections(image, detections)
    
    # Add template in top-left corner
    h, w = result.shape[:2]
    y_offset, x_offset = 10, 10
    
    if y_offset + new_th < h and x_offset + new_tw < w:
        # Create background for template
        cv2.rectangle(result, 
                     (x_offset - 5, y_offset - 5),
                     (x_offset + new_tw + 5, y_offset + new_th + 25),
                     (0, 0, 0), -1)
        
        # Place template
        result[y_offset:y_offset+new_th, x_offset:x_offset+new_tw] = template_resized
        
        # Add label
        cv2.putText(result, "Template", 
                   (x_offset, y_offset + new_th + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def create_processing_stages_visualization(images: Dict[str, np.ndarray],
                                         stage_names: List[str]) -> np.ndarray:
    """
    Create visualization showing different processing stages
    
    Args:
        images: Dictionary mapping stage names to images
        stage_names: Ordered list of stage names
        
    Returns:
        Combined visualization of processing stages
    """
    if not images or not stage_names:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Calculate grid size
    n_stages = len(stage_names)
    cols = min(3, n_stages)
    rows = (n_stages + cols - 1) // cols
    
    # Find target size for each image
    first_img = list(images.values())[0]
    target_h, target_w = 200, 200
    
    # Create result image
    result_h = rows * (target_h + 30) + 30
    result_w = cols * (target_w + 20) + 20
    result = np.zeros((result_h, result_w, 3), dtype=np.uint8)
    
    for i, stage_name in enumerate(stage_names):
        if stage_name not in images:
            continue
        
        row = i // cols
        col = i % cols
        
        # Position for this stage
        y = row * (target_h + 30) + 30
        x = col * (target_w + 20) + 10
        
        # Resize and convert image
        stage_img = images[stage_name]
        
        if len(stage_img.shape) == 2:
            stage_img = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
        
        stage_img_resized = cv2.resize(stage_img, (target_w, target_h))
        
        # Place image
        result[y:y+target_h, x:x+target_w] = stage_img_resized
        
        # Add title
        cv2.putText(result, stage_name,
                   (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result


def save_visualization_to_buffer(image: np.ndarray, format: str = 'PNG') -> io.BytesIO:
    """
    Save visualization to memory buffer
    
    Args:
        image: Image to save
        format: Image format ('PNG', 'JPEG', etc.)
        
    Returns:
        Memory buffer containing image data
    """
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis('off')
    
    # Save to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format.lower(), bbox_inches='tight', dpi=150)
    buffer.seek(0)
    
    plt.close(fig)
    
    return buffer


def create_detection_summary_plot(results: DetectionResults) -> io.BytesIO:
    """
    Create summary plot of detection results
    
    Args:
        results: Detection results
        
    Returns:
        Memory buffer containing plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    if not results.detections:
        fig.suptitle("No Detections Found")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plt.close(fig)
        return buffer
    
    # Extract data
    confidences = [det.confidence for det in results.detections]
    areas = [det.area for det in results.detections]
    scales = [det.scale for det in results.detections]
    angles = [det.angle for det in results.detections]
    
    # Confidence histogram
    ax1.hist(confidences, bins=20, alpha=0.7, color='blue')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Area histogram
    ax2.hist(areas, bins=20, alpha=0.7, color='green')
    ax2.set_xlabel('Area (pixels²)')
    ax2.set_ylabel('Count')
    ax2.set_title('Detection Area Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Scale vs Confidence scatter
    ax3.scatter(scales, confidences, alpha=0.6, c='red')
    ax3.set_xlabel('Scale')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Scale vs Confidence')
    ax3.grid(True, alpha=0.3)
    
    # Angle histogram
    ax4.hist(angles, bins=20, alpha=0.7, color='orange')
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Count')
    ax4.set_title('Angle Distribution')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Detection Results Summary - {len(results.detections)} detections')
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    plt.close(fig)
    
    return buffer
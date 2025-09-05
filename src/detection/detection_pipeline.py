#!/usr/bin/env python3
"""
Detection Pipeline for Analytical Symbol Detection System
Orchestrates the complete symbol detection workflow
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import os
import sys

# Add the project root to the path to import the original analytical_symbol_detection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.config_manager import DetectionConfig, ConfigurationManager
from src.detection.data_classes import (
    Detection, DetectionResults, TemplateMatchCandidate, 
    ProcessingStats, PipelineResults
)
from src.detection.template_extractor import TemplateExtractor
from src.processors.image_processor import ImagePreprocessor


class TemplateMatchingEngine:
    """Core template matching engine using existing analytical_symbol_detection logic"""
    
    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED):
        """Initialize template matching engine"""
        self.method = method
    
    def rotate_bound(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image keeping all content - from analytical_symbol_detection.py"""
        (h, w) = img.shape[:2]
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def local_peaks(self, resmap: np.ndarray, thresh: float, max_peaks: int) -> List[Tuple[int, int, float]]:
        """Return (x,y,score) local maxima above thresh - from analytical_symbol_detection.py"""
        if resmap.size == 0:
            return []
        
        peaks = []
        r = resmap.astype(np.float32)
        
        # Find local maxima
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dil = cv2.dilate(r, k)
        mask = (r == dil) & (r >= thresh)
        ys, xs = np.where(mask)
        
        for y, x in zip(ys, xs):
            peaks.append((int(x), int(y), float(r[y, x])))
        
        peaks.sort(key=lambda t: t[2], reverse=True)
        return peaks[:max_peaks]
    
    def match_coarse(self, search_small: np.ndarray,
                    tmpl_base: np.ndarray,
                    scales: List[float],
                    angles: List[float],
                    method: int,
                    thresh: float,
                    max_keep: int,
                    shrink: float) -> List[TemplateMatchCandidate]:
        """
        Coarse matching stage - adapted from analytical_symbol_detection.py
        
        Returns candidates in full-resolution coordinates
        """
        Hs, Ws = search_small.shape[:2]
        candidates = []
        
        for ang in angles:
            tmpl_rot = self.rotate_bound(tmpl_base, ang) if abs(ang) > 1e-3 else tmpl_base
            th0, tw0 = tmpl_rot.shape[:2]
            
            for s in scales:
                tw = max(8, int(round(tw0 * s * shrink)))
                th = max(8, int(round(th0 * s * shrink)))
                tmpl_rs = cv2.resize(tmpl_rot, (tw, th), interpolation=cv2.INTER_AREA)
                
                if tw >= Ws or th >= Hs:
                    continue
                
                res = cv2.matchTemplate(search_small, tmpl_rs, method)
                
                # For SQDIFF methods, convert to similarity
                if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    res_sim = 1.0 - res
                    peaks = self.local_peaks(res_sim, thresh, max_keep)
                else:
                    peaks = self.local_peaks(res, thresh, max_keep)
                
                for x, y, sc in peaks:
                    # Map back to full-res coordinates
                    x1 = int(round(x / shrink))
                    y1 = int(round(y / shrink))
                    x2 = int(round((x + tw) / shrink))
                    y2 = int(round((y + th) / shrink))
                    
                    candidate = TemplateMatchCandidate(
                        x=x1, y=y1,
                        width=x2-x1, height=y2-y1,
                        score=sc, scale=s, angle=ang
                    )
                    candidates.append(candidate)
        
        # Keep global top-K by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_keep]
    
    def refine_one(self, full_img: np.ndarray,
                  tmpl_full_rot_scaled: np.ndarray,
                  candidate: TemplateMatchCandidate,
                  method: int,
                  pad_ratio: float,
                  thresh: float) -> Optional[TemplateMatchCandidate]:
        """Refine a coarse candidate - adapted from analytical_symbol_detection.py"""
        H, W = full_img.shape[:2]
        th, tw = tmpl_full_rot_scaled.shape[:2]
        
        # Expand by pad_ratio of template size
        pad_w = int(round(tw * pad_ratio))
        pad_h = int(round(th * pad_ratio))
        cx = candidate.x + candidate.width // 2
        cy = candidate.y + candidate.height // 2
        
        rx1 = max(0, cx - tw//2 - pad_w)
        ry1 = max(0, cy - th//2 - pad_h)
        rx2 = min(W, cx + tw//2 + pad_w)
        ry2 = min(H, cy + th//2 + pad_h)
        
        roi = full_img[ry1:ry2, rx1:rx2]
        
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None
        
        res = cv2.matchTemplate(roi, tmpl_full_rot_scaled, method)
        
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            sc = 1.0 - min_val
            best = min_loc
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            sc = max_val
            best = max_loc
        
        if sc < thresh:
            return None
        
        bx = best[0] + rx1
        by = best[1] + ry1
        
        return TemplateMatchCandidate(
            x=bx, y=by,
            width=tw, height=th,
            score=sc, scale=candidate.scale, angle=candidate.angle
        )
    
    def apply_nms(self, candidates: List[TemplateMatchCandidate], 
                 iou_threshold: float = 0.3) -> List[TemplateMatchCandidate]:
        """Apply Non-Maximum Suppression - adapted from analytical_symbol_detection.py"""
        if not candidates:
            return []
        
        # Convert to format expected by NMS
        boxes = [[c.x1, c.y1, c.x2, c.y2] for c in candidates]
        scores = [c.score for c in candidates]
        
        keep_indices = self.nms_xyxy(boxes, scores, iou_threshold)
        return [candidates[i] for i in keep_indices]
    
    def nms_xyxy(self, boxes: List[List[int]], scores: List[float], iou_thresh=0.3) -> List[int]:
        """NMS implementation - from analytical_symbol_detection.py"""
        if not boxes:
            return []
        
        b = np.array(boxes, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        x1, y1, x2, y2 = b.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w_ = np.maximum(0.0, xx2 - xx1 + 1)
            h_ = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w_ * h_
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        
        return keep


class DetectionPipeline:
    """Main detection pipeline orchestrator"""
    
    def __init__(self, config: DetectionConfig):
        """Initialize detection pipeline"""
        self.config = config
        self.template_extractor = TemplateExtractor()
        self.image_processor = ImagePreprocessor()
        self.matching_engine = TemplateMatchingEngine()
        self.processing_stats = []
    
    def run(self, image: np.ndarray, template: np.ndarray) -> PipelineResults:
        """
        Run complete detection pipeline
        
        Args:
            image: Full input image
            template: Template for detection
            
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        self.processing_stats = []
        
        try:
            # Stage 1: Image preprocessing
            stage_stats = ProcessingStats("image_preprocessing")
            processed_image = self._preprocess_image(image)
            stage_stats.finish()
            self.processing_stats.append(stage_stats)
            
            # Stage 2: Template preprocessing
            stage_stats = ProcessingStats("template_preprocessing")
            processed_template = self._preprocess_template(template)
            template_metadata = self.template_extractor.analyze_template_features(template)
            stage_stats.finish()
            self.processing_stats.append(stage_stats)
            
            # Stage 3: Coarse matching
            stage_stats = ProcessingStats("coarse_matching")
            coarse_candidates = self._run_coarse_matching(processed_image, processed_template)
            stage_stats.additional_info['num_candidates'] = len(coarse_candidates)
            stage_stats.finish()
            self.processing_stats.append(stage_stats)
            
            # Stage 4: Refinement
            stage_stats = ProcessingStats("refinement")
            refined_candidates = self._refine_candidates(
                processed_image, processed_template, coarse_candidates
            )
            stage_stats.additional_info['num_refined'] = len(refined_candidates)
            stage_stats.finish()
            self.processing_stats.append(stage_stats)
            
            # Stage 5: Non-Maximum Suppression
            stage_stats = ProcessingStats("nms")
            final_candidates = self.matching_engine.apply_nms(refined_candidates, iou_threshold=0.25)
            stage_stats.additional_info['num_final'] = len(final_candidates)
            stage_stats.finish()
            self.processing_stats.append(stage_stats)
            
            # Convert to detections
            detections = [
                candidate.to_detection(self.config.class_name) 
                for candidate in final_candidates
            ]
            
            total_time = time.time() - start_time
            
            # Create results
            detection_results = DetectionResults(
                detections=detections,
                processing_time=total_time,
                metadata={
                    'image_shape': image.shape,
                    'template_shape': template.shape,
                    'config_used': self.config.to_dict(),
                    'num_scales': len(self.config.scales),
                    'num_angles': len(self.config.angles),
                    'use_edges': self.config.use_edges
                }
            )
            
            return PipelineResults(
                detection_results=detection_results,
                processing_stats=self.processing_stats,
                template_metadata=template_metadata,
                config_used=self.config.to_dict(),
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            return PipelineResults(
                detection_results=DetectionResults([], total_time),
                processing_stats=self.processing_stats,
                template_metadata={},
                config_used=self.config.to_dict(),
                success=False,
                error_message=str(e)
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess full image for detection"""
        if self.config.use_edges:
            # Convert to grayscale first if needed
            if len(image.shape) == 3:
                gray = self.image_processor.convert_to_grayscale(image)
            else:
                gray = image.copy()
            
            # Extract edges
            return self.image_processor.extract_edges(gray)
        else:
            # Use grayscale
            if len(image.shape) == 3:
                return self.image_processor.convert_to_grayscale(image)
            else:
                return image.copy()
    
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess template for detection"""
        return self.template_extractor.preprocess_template(
            template, 
            use_edges=self.config.use_edges,
            enhance_contrast=True,
            denoise=True
        )
    
    def _run_coarse_matching(self, image: np.ndarray, template: np.ndarray) -> List[TemplateMatchCandidate]:
        """Run coarse matching stage"""
        # Downscale image for coarse search
        shrink = self.config.coarse_scale
        if shrink < 1.0:
            height, width = image.shape[:2]
            search_small = cv2.resize(
                image, 
                (int(width * shrink), int(height * shrink)), 
                interpolation=cv2.INTER_AREA
            )
        else:
            search_small = image.copy()
        
        # Convert method name to OpenCV constant
        method_map = {
            "CCOEFF": cv2.TM_CCOEFF,
            "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "SQDIFF": cv2.TM_SQDIFF,
            "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            "CCORR_NORMED": cv2.TM_CCORR_NORMED,
        }
        method = method_map.get(self.config.method, cv2.TM_CCOEFF_NORMED)
        
        return self.matching_engine.match_coarse(
            search_small=search_small,
            tmpl_base=template,
            scales=self.config.scales,
            angles=self.config.angles,
            method=method,
            thresh=self.config.threshold,
            max_keep=self.config.topk,
            shrink=shrink
        )
    
    def _refine_candidates(self, image: np.ndarray, template: np.ndarray, 
                          candidates: List[TemplateMatchCandidate]) -> List[TemplateMatchCandidate]:
        """Refine coarse candidates at full resolution"""
        refined = []
        
        # Precompute rotated+scaled templates for efficiency
        template_cache = {}
        
        # Convert method name to OpenCV constant
        method_map = {
            "CCOEFF": cv2.TM_CCOEFF,
            "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "SQDIFF": cv2.TM_SQDIFF,
            "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            "CCORR_NORMED": cv2.TM_CCORR_NORMED,
        }
        method = method_map.get(self.config.method, cv2.TM_CCOEFF_NORMED)
        
        for candidate in candidates:
            key = (candidate.scale, candidate.angle)
            
            if key not in template_cache:
                # Create rotated and scaled template
                if abs(candidate.angle) > 1e-3:
                    tmpl_rot = self.matching_engine.rotate_bound(template, candidate.angle)
                else:
                    tmpl_rot = template
                
                th0, tw0 = tmpl_rot.shape[:2]
                tw = max(8, int(round(tw0 * candidate.scale)))
                th = max(8, int(round(th0 * candidate.scale)))
                
                template_cache[key] = cv2.resize(
                    tmpl_rot, (tw, th), interpolation=cv2.INTER_AREA
                )
            
            refined_candidate = self.matching_engine.refine_one(
                full_img=image,
                tmpl_full_rot_scaled=template_cache[key],
                candidate=candidate,
                method=method,
                pad_ratio=self.config.refine_pad,
                thresh=self.config.threshold * 0.95  # Slight relaxation
            )
            
            if refined_candidate:
                refined.append(refined_candidate)
        
        return refined
    
    def export_yolo_labels(self, results: DetectionResults, 
                          image_width: int, image_height: int,
                          output_path: str, class_id: int = 0) -> bool:
        """
        Export detection results in YOLO format
        
        Args:
            results: Detection results
            image_width: Full image width
            image_height: Full image height
            output_path: Path to save labels
            class_id: Class ID for YOLO format
            
        Returns:
            True if exported successfully
        """
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            with open(output_path, 'w') as f:
                for detection in results.detections:
                    yolo_line = detection.to_yolo_format(image_width, image_height, class_id)
                    f.write(yolo_line + '\n')
            
            return True
            
        except Exception as e:
            print(f"Error exporting YOLO labels: {str(e)}")
            return False
    
    def save_annotated_image(self, image: np.ndarray, results: DetectionResults, 
                           output_path: str) -> bool:
        """
        Save image with detection annotations
        
        Args:
            image: Original image
            results: Detection results  
            output_path: Path to save annotated image
            
        Returns:
            True if saved successfully
        """
        try:
            annotated = image.copy()
            
            for detection in results.detections:
                # Draw bounding box
                cv2.rectangle(annotated, (detection.x1, detection.y1), 
                            (detection.x2, detection.y2), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"{detection.confidence:.2f}"
                cv2.putText(annotated, label, 
                          (detection.x1, max(10, detection.y1 - 5)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            cv2.imwrite(output_path, annotated)
            
            return True
            
        except Exception as e:
            print(f"Error saving annotated image: {str(e)}")
            return False
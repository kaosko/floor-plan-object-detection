#!/usr/bin/env python3
"""
Configuration Manager for Analytical Symbol Detection System
Handles all configuration parameters and persistence
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict, Any
import argparse


@dataclass
class DetectionConfig:
    """Configuration for symbol detection parameters"""
    # PDF settings
    pdf_path: str = ""
    page: int = 0
    zoom: float = 6.0
    
    # Detection parameters
    threshold: float = 0.65
    scales: List[float] = field(default_factory=lambda: [0.95, 1.0, 1.05])
    angles: List[float] = field(default_factory=lambda: [0.0])
    method: str = "CCOEFF_NORMED"
    class_name: str = "hvac"
    use_edges: bool = True
    
    # ROI settings
    roi: Optional[Tuple[int, int, int, int]] = None
    reuse_roi: bool = False
    preview_width: int = 2400
    preview_height: int = 1400
    
    # Performance settings
    coarse_scale: float = 0.5
    topk: int = 300
    refine_pad: float = 0.5
    
    # Output settings
    output_dir: str = "symbol_detection_output"
    save_debug_images: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate_config()
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Validate threshold
        if not (0.0 <= self.threshold <= 1.0):
            errors.append("Threshold must be between 0.0 and 1.0")
        
        # Validate zoom
        if self.zoom <= 0:
            errors.append("Zoom must be positive")
        
        # Validate coarse scale
        if not (0.0 < self.coarse_scale <= 1.0):
            errors.append("Coarse scale must be between 0.0 and 1.0")
        
        # Validate topk
        if self.topk < 1:
            errors.append("Top-K must be at least 1")
        
        # Validate refine_pad
        if self.refine_pad < 0:
            errors.append("Refinement padding must be non-negative")
        
        # Validate scales
        if not self.scales or any(s <= 0 for s in self.scales):
            errors.append("Scales must be positive and non-empty")
        
        # Validate method
        valid_methods = ['CCOEFF', 'CCOEFF_NORMED', 'SQDIFF', 'SQDIFF_NORMED', 'CCORR_NORMED']
        if self.method not in valid_methods:
            errors.append(f"Method must be one of: {valid_methods}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate_config()


class ConfigurationManager:
    """Manages configuration loading, saving, and validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.config = DetectionConfig()
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            self.config.from_dict(config_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {filepath}: {str(e)}")
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration to {filepath}: {str(e)}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        self.config.from_dict(config_dict)
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        return self.config.validate_config()
    
    def to_argparse_namespace(self) -> argparse.Namespace:
        """Convert configuration to argparse namespace for compatibility"""
        # Map configuration to argparse-style namespace
        args = argparse.Namespace()
        
        # Direct mappings
        args.pdf = self.config.pdf_path
        args.page = self.config.page
        args.zoom = self.config.zoom
        args.threshold = self.config.threshold
        args.class_name = self.config.class_name
        args.use_edges = self.config.use_edges
        args.roi = self.config.roi
        args.reuse_roi = self.config.reuse_roi
        args.preview_width = self.config.preview_width
        args.preview_height = self.config.preview_height
        args.coarse = self.config.coarse_scale
        args.topk = self.config.topk
        args.refine_pad = self.config.refine_pad
        args.outdir = self.config.output_dir
        
        # Convert method name to OpenCV constant
        import cv2
        method_map = {
            "CCOEFF": cv2.TM_CCOEFF,
            "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "SQDIFF": cv2.TM_SQDIFF,
            "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            "CCORR_NORMED": cv2.TM_CCORR_NORMED,
        }
        args.method = method_map.get(self.config.method, cv2.TM_CCOEFF_NORMED)
        
        # Convert lists to expected format
        args.scales = self.config.scales
        args.angles = self.config.angles
        
        return args
    
    def from_argparse_namespace(self, args: argparse.Namespace) -> None:
        """Update configuration from argparse namespace"""
        # Reverse mapping from argparse namespace
        config_dict = {}
        
        if hasattr(args, 'pdf'):
            config_dict['pdf_path'] = args.pdf
        if hasattr(args, 'page'):
            config_dict['page'] = args.page
        if hasattr(args, 'zoom'):
            config_dict['zoom'] = args.zoom
        if hasattr(args, 'threshold'):
            config_dict['threshold'] = args.threshold
        if hasattr(args, 'class_name'):
            config_dict['class_name'] = args.class_name
        if hasattr(args, 'use_edges'):
            config_dict['use_edges'] = args.use_edges
        if hasattr(args, 'roi'):
            config_dict['roi'] = args.roi
        if hasattr(args, 'reuse_roi'):
            config_dict['reuse_roi'] = args.reuse_roi
        if hasattr(args, 'preview_width'):
            config_dict['preview_width'] = args.preview_width
        if hasattr(args, 'preview_height'):
            config_dict['preview_height'] = args.preview_height
        if hasattr(args, 'coarse'):
            config_dict['coarse_scale'] = args.coarse
        if hasattr(args, 'topk'):
            config_dict['topk'] = args.topk
        if hasattr(args, 'refine_pad'):
            config_dict['refine_pad'] = args.refine_pad
        if hasattr(args, 'outdir'):
            config_dict['output_dir'] = args.outdir
        if hasattr(args, 'scales'):
            config_dict['scales'] = args.scales
        if hasattr(args, 'angles'):
            config_dict['angles'] = args.angles
        
        # Convert OpenCV method constant back to string
        if hasattr(args, 'method'):
            import cv2
            method_reverse_map = {
                cv2.TM_CCOEFF: "CCOEFF",
                cv2.TM_CCOEFF_NORMED: "CCOEFF_NORMED",
                cv2.TM_SQDIFF: "SQDIFF",
                cv2.TM_SQDIFF_NORMED: "SQDIFF_NORMED",
                cv2.TM_CCORR_NORMED: "CCORR_NORMED",
            }
            config_dict['method'] = method_reverse_map.get(args.method, "CCOEFF_NORMED")
        
        self.config.from_dict(config_dict)
    
    def get_default_config(self) -> DetectionConfig:
        """Get default configuration"""
        return DetectionConfig()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = DetectionConfig()
    
    def create_output_directory(self) -> str:
        """Create output directory if it doesn't exist"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        return self.config.output_dir
    
    def get_roi_file_path(self) -> str:
        """Get path for ROI save file"""
        return os.path.join(self.config.output_dir, "roi.json")
    
    def save_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Save ROI to file"""
        roi_path = self.get_roi_file_path()
        os.makedirs(os.path.dirname(roi_path), exist_ok=True)
        with open(roi_path, 'w') as f:
            json.dump(list(roi), f)
        self.config.roi = roi
    
    def load_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """Load ROI from file"""
        roi_path = self.get_roi_file_path()
        if os.path.exists(roi_path):
            try:
                with open(roi_path, 'r') as f:
                    roi_list = json.load(f)
                roi = tuple(roi_list)
                self.config.roi = roi
                return roi
            except Exception:
                return None
        return None
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with keyword arguments"""
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        self.config.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.config.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"ConfigurationManager(config={self.config})"


# Utility functions for configuration management
def create_config_from_args(args: argparse.Namespace) -> ConfigurationManager:
    """Create configuration manager from argparse arguments"""
    config_manager = ConfigurationManager()
    config_manager.from_argparse_namespace(args)
    return config_manager


def merge_configs(base_config: DetectionConfig, update_config: Dict[str, Any]) -> DetectionConfig:
    """Merge base configuration with updates"""
    config_dict = base_config.to_dict()
    config_dict.update(update_config)
    
    new_config = DetectionConfig()
    new_config.from_dict(config_dict)
    return new_config
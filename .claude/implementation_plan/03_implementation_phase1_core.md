# Phase 1: Core Infrastructure Implementation

## Objective
Set up the foundational components and data structures needed by all other modules.

## Tasks

### Task 1.1: Project Setup
**File**: `src/__init__.py`, `requirements.txt`, `setup.py`

**Requirements**:
- Create project directory structure
- Set up virtual environment
- Install dependencies
- Configure logging

**Dependencies**:
```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
PyMuPDF>=1.23.0
Pillow>=10.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

**Test First**:
```python
# tests/test_project_setup.py
def test_imports():
    import cv2
    import numpy as np
    import fitz
    import streamlit
    assert True

def test_project_structure():
    import os
    assert os.path.exists('src')
    assert os.path.exists('tests')
    assert os.path.exists('data')
```

### Task 1.2: Data Classes Implementation
**File**: `src/models/data_models.py`

**Implementation**:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

@dataclass
class Detection:
    """Represents a single detection result"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    scale: float = 1.0
    angle: float = 0.0
    tile_id: Optional[int] = None
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

@dataclass
class DetectionConfig:
    """Configuration for detection pipeline"""
    pdf_path: str
    page: int = 0
    zoom: float = 6.0
    threshold: float = 0.65
    scales: List[float] = field(default_factory=lambda: [0.95, 1.0, 1.05])
    angles: List[float] = field(default_factory=lambda: [0])
    method: str = 'CCOEFF_NORMED'
    class_name: str = 'object'
    use_edges: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None
    coarse_scale: float = 0.5
    topk: int = 300
    refine_pad: float = 0.5
    output_dir: str = 'output'
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.zoom <= 0:
            raise ValueError("Zoom must be positive")
        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not 0 < self.coarse_scale <= 1:
            raise ValueError("Coarse scale must be between 0 and 1")
        return True

@dataclass
class DetectionResults:
    """Container for detection results"""
    detections: List[Detection]
    processing_time: float
    metadata: Dict = field(default_factory=dict)
    
    def filter_by_confidence(self, min_conf: float) -> 'DetectionResults':
        filtered = [d for d in self.detections if d.confidence >= min_conf]
        return DetectionResults(filtered, self.processing_time, self.metadata)
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionResults':
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionResults(filtered, self.processing_time, self.metadata)
```

**Tests**:
```python
# tests/test_data_models.py
import pytest
from src.models.data_models import Detection, DetectionConfig, DetectionResults

def test_detection_creation():
    det = Detection(10, 20, 100, 200, 0.95, 'door')
    assert det.bbox == (10, 20, 100, 200)
    assert det.center == (55, 110)
    assert det.area == 90 * 180

def test_detection_config_validation():
    config = DetectionConfig(pdf_path='test.pdf')
    assert config.validate()
    
    with pytest.raises(ValueError):
        bad_config = DetectionConfig(pdf_path='test.pdf', zoom=-1)
        bad_config.validate()

def test_detection_results_filtering():
    detections = [
        Detection(0, 0, 10, 10, 0.9, 'door'),
        Detection(20, 20, 30, 30, 0.5, 'window'),
        Detection(40, 40, 50, 50, 0.8, 'door')
    ]
    results = DetectionResults(detections, 1.5)
    
    filtered = results.filter_by_confidence(0.7)
    assert len(filtered.detections) == 2
    
    filtered = results.filter_by_class(['window'])
    assert len(filtered.detections) == 1
```

### Task 1.3: Configuration Manager
**File**: `src/core/config_manager.py`

**Implementation**:
```python
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from src.models.data_models import DetectionConfig

class ConfigurationManager:
    """Manages application configuration"""
    
    def __init__(self):
        self.config = DetectionConfig(pdf_path='')
        
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        self.load_from_dict(config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'pdf_path': self.config.pdf_path,
            'page': self.config.page,
            'zoom': self.config.zoom,
            'threshold': self.config.threshold,
            'scales': self.config.scales,
            'angles': self.config.angles,
            'method': self.config.method,
            'class_name': self.config.class_name,
            'use_edges': self.config.use_edges,
            'roi': self.config.roi,
            'coarse_scale': self.config.coarse_scale,
            'topk': self.config.topk,
            'refine_pad': self.config.refine_pad,
            'output_dir': self.config.output_dir
        }
    
    def to_argparse_namespace(self) -> argparse.Namespace:
        """Convert to argparse namespace for backward compatibility"""
        return argparse.Namespace(**self.to_dict())
    
    def from_argparse_args(self, args: argparse.Namespace) -> None:
        """Load from argparse arguments"""
        self.load_from_dict(vars(args))
```

**Tests**:
```python
# tests/test_config_manager.py
import pytest
import json
import tempfile
from src.core.config_manager import ConfigurationManager

def test_config_manager_dict_operations():
    manager = ConfigurationManager()
    config_dict = {
        'pdf_path': 'test.pdf',
        'zoom': 8.0,
        'threshold': 0.7
    }
    manager.load_from_dict(config_dict)
    
    assert manager.config.pdf_path == 'test.pdf'
    assert manager.config.zoom == 8.0
    assert manager.config.threshold == 0.7

def test_config_manager_file_operations():
    manager = ConfigurationManager()
    manager.config.pdf_path = 'test.pdf'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        manager.save_to_file(f.name)
        
        # Load from saved file
        new_manager = ConfigurationManager()
        new_manager.load_from_file(f.name)
        
        assert new_manager.config.pdf_path == 'test.pdf'
```

## Deliverables
1. Complete project structure with directories
2. Data models with validation
3. Configuration manager with persistence
4. All tests passing with >90% coverage
5. Basic logging configuration

## Success Criteria
- All unit tests pass
- Configuration can be loaded/saved
- Data models properly validate input
- Project can be imported without errors
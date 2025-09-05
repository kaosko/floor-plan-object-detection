# Class Architecture Design

## Core Classes

### 1. PDFProcessor
**Responsibility**: Handle PDF to PNG conversion
```python
class PDFProcessor:
    def __init__(self, pdf_path: str, page: int = 0, zoom: float = 6.0)
    def render_to_png(self, output_dir: str) -> str
    def get_page_count(self) -> int
    def get_page_dimensions(self, page: int) -> Tuple[float, float]
```

### 2. ImagePreprocessor
**Responsibility**: Image preprocessing operations
```python
class ImagePreprocessor:
    def __init__(self)
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray
    def extract_edges(self, image: np.ndarray) -> np.ndarray
    def apply_clahe(self, image: np.ndarray) -> np.ndarray
    def binarize(self, image: np.ndarray) -> np.ndarray
```

### 3. ROISelector
**Responsibility**: Region of Interest selection and management
```python
class ROISelector:
    def __init__(self)
    def select_interactive(self, image: np.ndarray) -> Tuple[int, int, int, int]
    def save_roi(self, roi: Tuple, filepath: str)
    def load_roi(self, filepath: str) -> Tuple[int, int, int, int]
    def validate_roi(self, roi: Tuple, image_shape: Tuple) -> Tuple[int, int, int, int]
```

### 4. TemplateExtractor
**Responsibility**: Extract and analyze templates from ROI
```python
class TemplateExtractor:
    def __init__(self)
    def extract_template(self, image: np.ndarray, roi: Tuple) -> np.ndarray
    def create_interior_mask(self, template: np.ndarray) -> np.ndarray
    def tighten_roi(self, roi: Tuple, mask: np.ndarray) -> Tuple
    def analyze_template_features(self, template: np.ndarray) -> Dict
```

### 5. BaseSymbolDetector (Abstract)
**Responsibility**: Base class for all symbol detectors
```python
class BaseSymbolDetector(ABC):
    def __init__(self, class_name: str, threshold: float = 0.65)
    @abstractmethod
    def detect(self, image: np.ndarray, template: np.ndarray) -> List[Detection]
    def apply_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]
    def export_yolo_format(self, detections: List[Detection], output_path: str)
```

### 6. Symbol-Specific Detectors
**Inherit from BaseSymbolDetector**
```python
class HVACSymbolDetector(BaseSymbolDetector):
    def __init__(self)
    def detect(self, image: np.ndarray, template: np.ndarray) -> List[Detection]

class DoorSymbolDetector(BaseSymbolDetector):
    def __init__(self)
    def detect(self, image: np.ndarray, template: np.ndarray) -> List[Detection]

class WindowSymbolDetector(BaseSymbolDetector):
    def __init__(self)
    def detect(self, image: np.ndarray, template: np.ndarray) -> List[Detection]

class ElectricalSymbolDetector(BaseSymbolDetector):
    def __init__(self)
    def detect(self, image: np.ndarray, template: np.ndarray) -> List[Detection]
```

### 7. TemplateMatchingEngine
**Responsibility**: Core template matching operations
```python
class TemplateMatchingEngine:
    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED)
    def match_coarse(self, image: np.ndarray, template: np.ndarray, params: Dict) -> List[Detection]
    def refine_matches(self, image: np.ndarray, template: np.ndarray, coarse_matches: List) -> List[Detection]
    def multi_scale_match(self, image: np.ndarray, template: np.ndarray, scales: List[float]) -> List[Detection]
    def multi_angle_match(self, image: np.ndarray, template: np.ndarray, angles: List[float]) -> List[Detection]
```

### 8. DetectionPipeline
**Responsibility**: Orchestrate the complete detection pipeline
```python
class DetectionPipeline:
    def __init__(self, config: DetectionConfig)
    def set_pdf_processor(self, processor: PDFProcessor)
    def set_preprocessor(self, preprocessor: ImagePreprocessor)
    def set_detector(self, detector: BaseSymbolDetector)
    def run(self) -> DetectionResults
    def save_results(self, results: DetectionResults, output_dir: str)
```

### 9. ConfigurationManager
**Responsibility**: Manage all configuration parameters
```python
class ConfigurationManager:
    def __init__(self)
    def load_from_dict(self, config_dict: Dict)
    def load_from_file(self, filepath: str)
    def save_to_file(self, filepath: str)
    def validate_config(self) -> bool
    def to_argparse_namespace(self) -> argparse.Namespace
```

### 10. VisualizationHelper
**Responsibility**: Handle all visualization tasks
```python
class VisualizationHelper:
    def __init__(self)
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray
    def create_tile_overlay(self, image: np.ndarray, tile_info: Dict) -> np.ndarray
    def visualize_template(self, template: np.ndarray) -> np.ndarray
    def create_debug_visualization(self, data: Dict) -> np.ndarray
```

## Data Classes

```python
@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str
    scale: float = 1.0
    angle: float = 0.0

@dataclass
class DetectionConfig:
    pdf_path: str
    page: int
    zoom: float
    threshold: float
    scales: List[float]
    angles: List[float]
    method: str
    class_name: str
    use_edges: bool
    roi: Optional[Tuple[int, int, int, int]]
    coarse_scale: float
    topk: int
    refine_pad: float

@dataclass
class DetectionResults:
    detections: List[Detection]
    processing_time: float
    metadata: Dict
```
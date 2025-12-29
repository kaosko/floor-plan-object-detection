"""
train_gpu.py - Train YOLOv11 with GPU
"""
import os
# CRITICAL: Set BEFORE any torch import
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
os.environ['TORCH_CUDA_MEMORY_ALLOCATOR'] = 'native'

import warnings
warnings.filterwarnings('ignore')

# Try to disable NVML at C level
import ctypes
try:
    # Preload a dummy to prevent NVML loading
    pass
except:
    pass

from ultralytics import YOLO
import torch

def main():
    # ===== CONFIGURATION =====
    DATASET_PATH = "./datasets/Floor_plan_multiple-1"
    DATA_YAML = f"{DATASET_PATH}/data.yaml"

    # ===== GPU CHECK =====
    print("="*60)
    print("GPU Check")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"   VRAM: {gpu_mem:.1f} GB")
        batch_size = 8
        device = 0
    else:
        print("⚠️  No GPU - using CPU")
        batch_size = 4
        device = 'cpu'

    # ===== VERIFY DATASET =====
    print("\n" + "="*60)
    print("Dataset Check")
    print("="*60)

    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset not found at: {DATA_YAML}")
        exit(1)

    import yaml
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✅ Dataset found: {DATASET_PATH}")
    print(f"\n📋 Classes ({len(config['names'])}):")
    for idx, name in enumerate(config['names']):
        print(f"  {idx}: {name}")

    # ===== TRAINING =====
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    model = YOLO('yolo11m.pt')

    print(f"\n🚀 Training with:")
    print(f"   Model: YOLOv11m")
    print(f"   Epochs: 100")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: 640")

    # Train with workers=0 to avoid multiprocessing issues on Python 3.14
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        batch=batch_size,
        imgsz=640,
        project='runs/train',
        name='floor_plan_2door',
        
        # Floor plan optimized
        degrees=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        
        # Device settings - disable features that need NVML
        device=device,
        workers=0,        # Disable multiprocessing workers
        amp=False,        # Disable AMP (avoids NVML memory queries)
        cache=True,       # Cache images for speed
        
        # Training
        patience=50,
        save=True,
        save_period=10,
        plots=True,
        optimizer='AdamW',
        lr0=0.001,
    )

    # ===== DONE =====
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)

    best_model = 'runs/train/floor_plan_2door/weights/best.pt'
    print(f"\nBest model saved to: {best_model}")

    # Verify
    final_model = YOLO(best_model)
    print(f"\n📋 Final Model Classes ({len(final_model.names)}):")
    for idx, name in final_model.names.items():
        print(f"  {idx}: {name}")


if __name__ == '__main__':
    # Required for multiprocessing on Windows/Python 3.14
    import multiprocessing
    multiprocessing.freeze_support()
    main()

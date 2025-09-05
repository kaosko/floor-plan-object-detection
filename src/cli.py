#!/usr/bin/env python3
"""
Command-line interface for analytical symbol detection.
Provides argparse compatibility for the Streamlit application.
"""

import argparse
import sys
import json
from pathlib import Path

from src.models.data_models import DetectionConfig
from src.detection.detection_pipeline import DetectionPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analytical Symbol Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--pdf", 
        required=True,
        help="Path to input PDF file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--page", 
        type=int, 
        default=0,
        help="Zero-based page index"
    )
    
    parser.add_argument(
        "--zoom", 
        type=float, 
        default=6.0,
        help="PDF rasterization zoom factor"
    )
    
    parser.add_argument(
        "--outdir", 
        default="detection_output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.65,
        help="Detection threshold (0-1)"
    )
    
    parser.add_argument(
        "--scales", 
        default="0.95,1.0,1.05",
        help="Comma-separated scale factors"
    )
    
    parser.add_argument(
        "--angles", 
        default="0",
        help="Comma-separated angles in degrees"
    )
    
    parser.add_argument(
        "--method",
        choices=['CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'],
        default="CCOEFF_NORMED",
        help="Template matching method"
    )
    
    parser.add_argument(
        "--class-name", 
        default="object",
        help="Symbol class name"
    )
    
    parser.add_argument(
        "--no-edges",
        dest="use_edges",
        action="store_false",
        help="Disable edge detection"
    )
    
    parser.add_argument(
        "--roi",
        type=str,
        help="ROI as x,y,w,h (skip interactive selection)"
    )
    
    parser.add_argument(
        "--reuse-roi",
        action="store_true",
        help="Reuse saved ROI if available"
    )
    
    parser.add_argument(
        "--coarse",
        type=float,
        default=0.5,
        help="Coarse search scale (0-1)"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Top-K coarse candidates"
    )
    
    parser.add_argument(
        "--refine-pad",
        type=float,
        default=0.5,
        help="Refinement padding ratio"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Streamlit GUI instead of CLI"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from JSON file"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to JSON file"
    )
    
    return parser.parse_args()

def args_to_config(args) -> DetectionConfig:
    """Convert argparse namespace to DetectionConfig"""
    # Parse scales and angles
    scales = [float(s.strip()) for s in args.scales.split(',') if s.strip()]
    angles = [float(a.strip()) for a in args.angles.split(',') if a.strip()]
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        parts = [int(v) for v in args.roi.split(',')]
        if len(parts) == 4:
            roi = tuple(parts)
    
    config = DetectionConfig(
        pdf_path=args.pdf,
        page=args.page,
        zoom=args.zoom,
        threshold=args.threshold,
        scales=scales,
        angles=angles,
        method=args.method,
        class_name=args.class_name,
        use_edges=args.use_edges,
        roi=roi,
        coarse_scale=args.coarse,
        topk=args.topk,
        refine_pad=args.refine_pad,
        output_dir=args.outdir
    )
    
    return config

def main():
    args = parse_args()
    
    # Launch GUI if requested
    if args.gui:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "src/ui/app.py"]
        sys.exit(stcli.main())
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DetectionConfig(**config_dict)
    else:
        config = args_to_config(args)
    
    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Save config if requested
    if args.save_config:
        config.save_to_file(args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Run detection pipeline
    print("Starting detection pipeline...")
    print(f"PDF: {config.pdf_path}")
    print(f"Page: {config.page}")
    print(f"Class: {config.class_name}")
    print(f"Output: {config.output_dir}")
    print()
    
    pipeline = DetectionPipeline(config)
    
    try:
        results = pipeline.run()
        
        print(f"\nDetection complete!")
        print(f"Found {len(results.detections)} instances")
        print(f"Processing time: {results.processing_time:.2f}s")
        
        # Save results
        output_files = pipeline.save_results(results)
        
        print("\nOutput files:")
        for file_type, filepath in output_files.items():
            print(f"  {file_type}: {filepath}")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
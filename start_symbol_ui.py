#!/usr/bin/env python3
"""
Startup script for Analytical Symbol Detection UI
Handles error checking and provides helpful startup messages
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'opencv-python', 
        'numpy',
        'pandas',
        'pymupdf',
        'matplotlib',
        'plotly',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            if package == 'opencv-python':
                try:
                    import cv2
                except ImportError:
                    missing_packages.append(package)
            elif package == 'pymupdf':
                try:
                    import fitz
                except ImportError:
                    missing_packages.append(package)
            else:
                missing_packages.append(package)
    
    return missing_packages

def main():
    """Main startup function"""
    print("🔍 Analytical Symbol Detection UI")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('analytical_symbol_detection.py'):
        print("❌ Error: analytical_symbol_detection.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check required packages
    print("🔍 Checking required packages...")
    missing = check_requirements()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
        
        # Check if we have a requirements file
        if os.path.exists('requirements.txt'):
            print("\nOr install all requirements:")
            print("pip install -r requirements.txt")
        
        sys.exit(1)
    
    print("✅ All required packages found")
    
    # Check if analytical symbol detection script exists and is valid
    print("🔍 Checking core detection script...")
    try:
        with open('analytical_symbol_detection.py', 'r') as f:
            content = f.read()
            if 'def main()' in content and 'argparse' in content:
                print("✅ Core detection script found and valid")
            else:
                print("⚠️  Warning: Core detection script may be incomplete")
    except Exception as e:
        print(f"❌ Error reading detection script: {e}")
        sys.exit(1)
    
    # Launch Streamlit
    print("🚀 Starting Symbol Detection UI...")
    print("\nThe interface will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Start streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'symbol_detection_ui.py']
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found!")
        print("Install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Installation script for ComfyUI Qwen-Image Plugin

This script handles the installation of dependencies and initial setup
for the Qwen-Image ComfyUI plugin.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def is_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def check_gpu_support():
    """Check for GPU support and recommend appropriate torch version."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No CUDA GPU detected. CPU mode will be used.")
            return False
    except ImportError:
        print("⚠ PyTorch not found. Will install CPU version.")
        return False

def install_requirements():
    """Install required packages."""
    print("Installing ComfyUI Qwen-Image Plugin dependencies...")
    
    # Read requirements file
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Check for GPU support first
    has_gpu = check_gpu_support()
    
    # Install requirements
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file)
        ])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\nVerifying installation...")
    
    required_packages = [
        "torch",
        "transformers", 
        "diffusers",
        "PIL",
        "numpy"
    ]
    
    all_good = True
    for package in required_packages:
        if package == "PIL":
            # PIL is imported as Pillow
            if is_package_installed("PIL"):
                print(f"✓ {package}")
            else:
                print(f"❌ {package}")
                all_good = False
        else:
            if is_package_installed(package):
                print(f"✓ {package}")
            else:
                print(f"❌ {package}")
                all_good = False
    
    return all_good

def download_model_info():
    """Provide information about model download."""
    print("\n" + "="*60)
    print("MODEL DOWNLOAD INFORMATION")
    print("="*60)
    print("The Qwen-Image model will be downloaded automatically on first use.")
    print("Model size: ~20GB")
    print("Download location: HuggingFace cache directory")
    print("\nTo pre-download the model, you can run:")
    print("python -c \"from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image')\"")
    print("\nOr use the model loader node in ComfyUI.")

def main():
    """Main installation function."""
    print("="*60)
    print("ComfyUI Qwen-Image Plugin Installation")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Install requirements
    if not install_requirements():
        print("❌ Installation failed!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("⚠ Some packages may not be installed correctly.")
        print("Please check the error messages above.")
    else:
        print("✓ All packages verified!")
    
    # Model download info
    download_model_info()
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    print("The ComfyUI Qwen-Image plugin is now ready to use.")
    print("Please restart ComfyUI to load the new nodes.")
    print("\nNodes available:")
    print("- 🎨 Qwen-Image Model Loader")
    print("- 🎨 Qwen-Image Generate")
    print("- 🎨 Qwen-Image Text Render")
    print("- 🎨 Qwen-Image Edit")
    print("- 🎨 Qwen-Image Understanding")
    print("\nFor usage examples, check the example_workflows/ directory.")
    print("For documentation, see README.md")

if __name__ == "__main__":
    main()

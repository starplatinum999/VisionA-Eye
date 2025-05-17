#!/usr/bin/env python3
"""
Test script to verify Vision AI installation and dependencies
"""

import os
import sys
import importlib.util
import subprocess
import shutil

def check_module(module_name):
    """Check if a Python module is installed"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} is NOT installed")
        return False
    else:
        print(f"✅ {module_name} is installed")
        return True

def check_command(command):
    """Check if a command is available"""
    path = shutil.which(command)
    if path is None:
        print(f"❌ {command} is NOT available")
        return False
    else:
        print(f"✅ {command} is available at {path}")
        return True

def check_directory(directory):
    """Check if a directory exists"""
    if os.path.isdir(directory):
        print(f"✅ Directory {directory} exists")
        return True
    else:
        print(f"❌ Directory {directory} does NOT exist")
        return False

def check_file(file_path):
    """Check if a file exists"""
    if os.path.isfile(file_path):
        print(f"✅ File {file_path} exists")
        return True
    else:
        print(f"❌ File {file_path} does NOT exist")
        return False

def main():
    """Run installation tests"""
    print("=" * 60)
    print("Vision AI Installation Test")
    print("=" * 60)
    
    # Check Python version
    py_version = sys.version.split()[0]
    if int(py_version.split(".")[0]) >= 3 and int(py_version.split(".")[1]) >= 8 or int(py_version.split(".")[0]) > 3:
        print(f"✅ Python version {py_version} is compatible")
    else:
        print(f"❌ Python version {py_version} is NOT compatible (3.8+ required)")
    
    # Check required modules
    required_modules = [
        "streamlit", "cv2", "numpy", "torch", "pandas", 
        "matplotlib", "plotly", "ultralytics", "llama_cpp"
    ]
    
    print("\nChecking required Python modules:")
    modules_missing = False
    for module in required_modules:
        if not check_module(module):
            modules_missing = True
    
    # Check project structure
    print("\nChecking project structure:")
    directories = [
        "app", 
        "app/components", 
        "app/components/detection",
        "app/components/tracking",
        "app/components/roi",
        "app/components/analytics",
        "app/components/symbolic_reasoning",
        "app/utils",
        "app/data",
        "app/models"
    ]
    
    structure_issues = False
    for directory in directories:
        if not check_directory(directory):
            structure_issues = True
    
    # Check core files
    print("\nChecking core files:")
    core_files = [
        "app.py",
        "requirements.txt",
        "app/components/detection/detector.py",
        "app/components/tracking/tracker.py",
        "app/components/roi/roi_manager.py",
        "app/components/analytics/analytics.py",
        "app/components/symbolic_reasoning/llm_reasoning.py",
        "app/utils/video_utils.py",
        "app/utils/download_models.py"
    ]
    
    files_missing = False
    for file in core_files:
        if not check_file(file):
            files_missing = True
    
    # Summary
    print("\n" + "=" * 60)
    print("Installation Test Summary")
    print("=" * 60)
    
    if modules_missing:
        print("❌ Some required Python modules are missing")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All required Python modules are installed")
    
    if structure_issues:
        print("❌ Project structure has issues")
    else:
        print("✅ Project structure is correct")
    
    if files_missing:
        print("❌ Some core files are missing")
    else:
        print("✅ All core files are present")
    
    # Check models
    model_files = [
        "app/models/yolo/yolov8n.pt",
        "app/models/deep_sort/ckpt.t7",
        "app/models/llm/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
    ]
    
    models_missing = False
    for model in model_files:
        if not os.path.exists(model):
            models_missing = True
            break
    
    if models_missing:
        print("❌ Models are not downloaded")
        print("   Run: python -m app.utils.download_models")
    else:
        print("✅ All required models are downloaded")
    
    # Final verdict
    print("\nFinal Verdict:")
    if not (modules_missing or structure_issues or files_missing):
        print("✅ Installation looks good! You can run the app with:")
        print("   streamlit run app.py")
    else:
        print("❌ There are issues with the installation that need to be fixed")

if __name__ == "__main__":
    main() 
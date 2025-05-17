#!/usr/bin/env python3

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_import(module_name, additional_info=""):
    """Check if a module can be imported and print status"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {module_name} - Version: {version} {additional_info}")
        return module
    except ImportError as e:
        print(f"❌ {module_name} - Import Error: {e}")
        return None
    except Exception as e:
        print(f"⚠️ {module_name} - Error: {e}")
        return None

def run_command(cmd):
    """Run a shell command and return its output"""
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.strip()}"

print("\n=== SYSTEM INFORMATION ===")
print(f"Python version: {sys.version}")
print(f"Executable path: {sys.executable}")
print(f"Platform: {sys.platform}")

# Check if we're using pyenv
pyenv_prefix = run_command("pyenv prefix 2>/dev/null || echo 'None'")
if pyenv_prefix != 'None':
    print(f"Using pyenv: {pyenv_prefix}")
    pyenv_version = run_command("pyenv version-name")
    print(f"Pyenv version: {pyenv_version}")

print("\n=== CHECKING CRITICAL LIBRARIES ===")
# Check core libraries
cv2 = check_import("cv2")
if cv2:
    print(f"OpenCV backend: {cv2.getBuildInformation().split('Use CUDA:')[1].split('\n')[0] if 'Use CUDA:' in cv2.getBuildInformation() else 'Unknown'}")

check_import("numpy")
check_import("PIL", "(Pillow)")

print("\n=== CHECKING STREAMLIT ===")
# Check streamlit
streamlit = check_import("streamlit")
if streamlit:
    streamlit_path = Path(streamlit.__file__).parent
    print(f"Streamlit path: {streamlit_path}")
else:
    print("Checking streamlit availability in PATH...")
    streamlit_path = run_command("which streamlit 2>/dev/null || echo 'Not found'")
    print(f"Streamlit executable: {streamlit_path}")

# Check streamlit_drawable_canvas
check_import("streamlit_drawable_canvas")

print("\n=== CHECKING ML LIBRARIES ===")
# Check ML libraries
check_import("torch")
check_import("onnx")
check_import("onnxruntime")

print("\n=== PATH ENVIRONMENT ===")
path = os.environ.get('PATH', '')
print(f"PATH: {path}")

print("\n=== PYTHON PATH ===")
for p in sys.path:
    print(f"  {p}")

print("\n=== DIAGNOSING POTENTIAL ISSUES ===")
# Check for potential issues
if 'pyenv' in path and 'pyenv' in pyenv_prefix:
    site_packages = run_command(f"{sys.executable} -c 'import site; print(site.getsitepackages()[0])'")
    print(f"Site packages: {site_packages}")
    
    pip_path = run_command(f"{sys.executable} -m pip --version")
    print(f"Pip path: {pip_path}")

print("\n=== RECOMMENDATIONS ===")
# Add recommendations
if streamlit and not check_import("streamlit_drawable_canvas", silent=True):
    print("Install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")

if not cv2:
    print("Install OpenCV: pip install opencv-python")

print("\nTo create a fresh virtual environment and install all dependencies:")
print("python -m venv venv")
print("source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
print("pip install -r requirements.txt")

if __name__ == "__main__":
    # Fix streamlit not in PATH
    if streamlit_path == "Not found":
        pip_bin = run_command(f"{sys.executable} -c 'import site; import os; print(os.path.join(site.USER_BASE, \"bin\"))'")
        print(f"\nAdd pip bin directory to PATH to find streamlit:")
        print(f"export PATH=\"{pip_bin}:$PATH\"") 
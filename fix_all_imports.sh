#!/bin/bash
# Comprehensive fix script for resolving all module import issues

echo "===================================================="
echo "Vision AI - Comprehensive Import Fix Script"
echo "===================================================="

# Get Python information
PYTHON_VERSION=$(python --version 2>&1)
PYTHON_PATH=$(which python)
PIP_PATH=$(which pip)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

echo "Python information:"
echo "Version: $PYTHON_VERSION"
echo "Path: $PYTHON_PATH"
echo "Pip: $PIP_PATH"
echo "Site-packages: $SITE_PACKAGES"
echo ""

# Check if this is a virtual environment
VIRTUAL_ENV_STATUS="Not in a virtual environment"
if [ -n "$VIRTUAL_ENV" ]; then
    VIRTUAL_ENV_STATUS="Active virtual environment: $VIRTUAL_ENV"
fi
echo "$VIRTUAL_ENV_STATUS"
echo ""

# Check Python imports path
echo "Python import paths:"
python -c "import sys; print('\n'.join(sys.path))"
echo ""

# Check pip list
echo "Currently installed packages:"
pip list
echo ""

# Fix 1: Create a dedicated virtual environment
echo "Creating a fresh virtual environment to isolate dependencies..."
python -m venv visionai_fresh_env

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source visionai_fresh_env/Scripts/activate
else
    source visionai_fresh_env/bin/activate
fi

echo "✅ Virtual environment created and activated"
echo "Python path now: $(which python)"

# Install all requirements
echo "Installing all requirements in the new environment..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "⚠️ Some packages could not be installed. Trying individual installations..."

    # Try installing packages individually
    declare -a packages=(
        "streamlit==1.31.0"
        "opencv-python==4.8.1.78"
        "ultralytics==8.0.196"
        "numpy==1.24.3"
        "pandas==2.1.0"
        "matplotlib==3.8.0"
        "torch==2.1.0"
        "torchvision==0.16.0"
        "pillow==10.0.1"
        "scikit-learn==1.3.0"
        "plotly==5.18.0"
        "streamlit-drawable-canvas==0.9.3"
        "supervision==0.18.0"
        "python-dotenv==1.0.0"
    )

    for package in "${packages[@]}"; do
        echo "Installing $package..."
        pip install $package
    done

    # Special handling for llama-cpp-python
    echo "Installing llama-cpp-python with optimizations for this system..."
    
    # Check if on Mac with Apple Silicon
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            echo "Detected Apple Silicon Mac, installing llama-cpp-python with Metal support..."
            pip uninstall -y llama-cpp-python
            CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python==0.2.19
        else
            pip install llama-cpp-python==0.2.19
        fi
    else
        pip install llama-cpp-python==0.2.19
    fi
fi

# Fix 2: Create app package helper
echo "Creating app package helper to ensure imports work..."
cat > app_fix.py << EOL
import sys
import os

# Get the absolute path to the current directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Add the base directory to the Python path if not already there
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
    print(f"Added {base_dir} to Python path")

# Import check
try:
    from app.components.detection.detector import YOLODetector
    from app.components.tracking.tracker import DeepSORTTracker
    from app.components.roi.roi_manager import ROIManager
    from app.components.analytics.analytics import AnalyticsManager
    from app.components.symbolic_reasoning.llm_reasoning import SymbolicReasoner
    from app.utils.video_utils import get_video_frame, process_video
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("Python path:")
for path in sys.path:
    print(f"  {path}")
EOL

# Test app imports
echo ""
echo "Testing app imports..."
python app_fix.py

# Create launch script
echo "Creating launch script..."
cat > launch.sh << EOL
#!/bin/bash
# Launch script for Vision AI

# Activate the virtual environment
if [[ "\$OSTYPE" == "msys" || "\$OSTYPE" == "win32" ]]; then
    source visionai_fresh_env/Scripts/activate
else
    source visionai_fresh_env/bin/activate
fi

# Run the application with the fixed Python path
PYTHONPATH=\$(pwd) streamlit run app.py
EOL

chmod +x launch.sh

# Final message
echo ""
echo "===================================================="
echo "Import Fix Complete! 🎉"
echo "===================================================="
echo ""
echo "To run Vision AI, use the following command:"
echo ""
echo "./launch.sh"
echo ""
echo "This will:"
echo "1. Activate the new virtual environment"
echo "2. Set the correct Python path"
echo "3. Run the application with streamlit"
echo "" 
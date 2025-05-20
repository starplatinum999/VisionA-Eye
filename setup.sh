#!/bin/bash
# Vision AI Setup Script

echo "===================================================="
echo "Vision AI - Smart Surveillance System Setup"
echo "===================================================="

# Check for Python
if command -v python3 &>/dev/null; then
    echo "✅ Python 3 found"
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    echo "✅ Python found"
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.8 or newer."
    exit 1
fi

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies directly, bypassing some problematic build systems
pip install --no-build-isolation PySide6 opencv-python numpy
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install llama-cpp-python --no-cache-dir

# Create model directories and download models manually
mkdir -p app/models/yolo
mkdir -p app/models/deep_sort
mkdir -p app/models/llm

# Download YOLOv8 model directly
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o app/models/yolo/yolov8n.pt

if [ $? -eq 0 ]; then
    echo "✅ Models downloaded successfully"
else
    echo "❌ Error downloading models"
    exit 1
fi

# Test installation
echo "Testing installation..."
$PYTHON_CMD test_installation.py

# Final message
echo ""
echo "===================================================="
echo "Setup Complete! 🎉"
echo "===================================================="
echo ""
echo "To run Vision AI, use the following command:"
echo ""
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "venv\\Scripts\\activate"
    else
        echo "source venv/bin/activate"
    fi
fi
echo "streamlit run app.py"
echo ""
echo "Thank you for installing Vision AI!" 
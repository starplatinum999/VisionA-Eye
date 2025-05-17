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

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p app/data
mkdir -p app/models/yolo
mkdir -p app/models/deep_sort
mkdir -p app/models/llm
echo "✅ Directories created"

# Download models
echo "Downloading models (this may take some time)..."
$PYTHON_CMD -m app.utils.download_models

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
#!/bin/bash
# Fix environment script for Vision AI

echo "===================================================="
echo "Vision AI - Environment Fix Script"
echo "===================================================="

# Identify Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Current Python version: $PYTHON_VERSION"

# Check if Streamlit is installed
if command -v streamlit &>/dev/null; then
    echo "✅ Streamlit is already in PATH"
else
    echo "❌ Streamlit not found in PATH"
    echo "Installing dependencies for current Python version..."
    
    # Install all requirements
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Error installing dependencies"
        exit 1
    fi
    
    # Check if streamlit is now available
    if command -v streamlit &>/dev/null; then
        echo "✅ Streamlit is now available"
    else
        echo "❌ Streamlit still not in PATH"
        echo "Adding pip's bin directory to PATH for this session..."
        
        # Try to find pip's bin directory
        PIP_BIN_DIR=$(python -c "import site; print(site.USER_BASE + '/bin')")
        export PATH="$PIP_BIN_DIR:$PATH"
        
        if command -v streamlit &>/dev/null; then
            echo "✅ Streamlit is now available after PATH update"
            echo "Consider adding this line to your ~/.bashrc or ~/.zshrc:"
            echo "export PATH=\"$PIP_BIN_DIR:\$PATH\""
        else
            echo "❌ Could not find streamlit"
            echo "You may need to manually locate streamlit and add it to your PATH"
        fi
    fi
fi

# Install missing packages if needed
echo "Checking for missing packages..."

# Check for plotly
python -c "import plotly.express" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing plotly..."
    pip install plotly==5.18.0
fi

# Check for streamlit-drawable-canvas
python -c "import streamlit_drawable_canvas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing streamlit-drawable-canvas..."
    pip install streamlit-drawable-canvas==0.9.3
fi

# Check for other common dependencies
python -c "import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pandas..."
    pip install pandas==2.1.0
fi

python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing numpy..."
    pip install numpy==1.24.3
fi

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing torch..."
    pip install torch==2.1.0 torchvision==0.16.0
fi

# Create virtual environment specifically for Vision AI (optional)
echo ""
read -p "Do you want to create a dedicated virtual environment for Vision AI? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python -m venv visionai_env
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source visionai_env/Scripts/activate
    else
        source visionai_env/bin/activate
    fi
    echo "✅ Virtual environment created and activated"
    
    # Install dependencies in the virtual environment
    echo "Installing dependencies in virtual environment..."
    pip install -r requirements.txt
    
    echo "To activate this environment in the future, run:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "source visionai_env/Scripts/activate"
    else
        echo "source visionai_env/bin/activate"
    fi
fi

# Test installation
echo ""
echo "Testing if streamlit is working..."
streamlit --version

# Test import required packages
echo ""
echo "Testing imports..."
python -c "import streamlit; import plotly.express; import numpy; import pandas; import torch; print('✅ All core imports successful')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some imports are still failing. Try running with the virtual environment option."
else
    echo "✅ All imports successful"
fi

# Final message
echo ""
echo "===================================================="
echo "Environment Fix Complete! 🎉"
echo "===================================================="
echo ""
echo "To run Vision AI, use the following command:"
echo ""
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "source visionai_env/Scripts/activate"
    else
        echo "source visionai_env/bin/activate"
    fi
fi
echo "streamlit run app.py"
echo "" 
#!/bin/bash
# Quick fix script for llama-cpp-python import issue

echo "Installing llama-cpp-python package..."
pip install llama-cpp-python==0.2.19

# Check if llama_cpp can be imported now
python -c "from llama_cpp import Llama" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ llama-cpp-python installed successfully!"
    echo "You can now run: streamlit run app.py"
else
    echo "❌ llama-cpp-python installation failed."
    echo "Running more detailed checks..."
    
    # Check if CMake is installed (required for llama-cpp-python)
    command -v cmake >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "CMake is not installed, which is required for building llama-cpp-python."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Please install CMake using Homebrew: brew install cmake"
        else
            echo "Please install CMake using your package manager"
        fi
    fi
    
    # Try installing with pip directly
    echo "Trying to reinstall with specific options..."
    pip uninstall -y llama-cpp-python
    pip install llama-cpp-python==0.2.19 --no-cache-dir
    
    # Check again
    python -c "from llama_cpp import Llama" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ llama-cpp-python reinstalled successfully!"
    else
        echo "❌ Still having issues with llama-cpp-python."
        echo "Please run the full fix script: ./fix_environment.sh"
        echo "You may need to install additional system dependencies."
    fi
fi 
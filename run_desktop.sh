#!/bin/bash

# Vision AI Desktop App Launch Script

# Find Python - check multiple possible paths on macOS
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif [ -f "/opt/homebrew/bin/python3" ]; then
    PYTHON_CMD="/opt/homebrew/bin/python3"
elif [ -f "/usr/local/bin/python3" ]; then
    PYTHON_CMD="/usr/local/bin/python3"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Activate virtual environment if it exists
if [ -d "visionai_fresh_env" ]; then
    echo "Activating virtual environment..."
    source visionai_fresh_env/bin/activate
    # When we activate a venv, we should use 'python' not 'python3'
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    fi
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    # When we activate a venv, we should use 'python' not 'python3'
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    fi
fi

echo "Python command: $PYTHON_CMD"

# Check if required dependencies are installed
echo "Checking dependencies..."

# Check PySide6
$PYTHON_CMD -c "import PySide6" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "PySide6 not found. Installing requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Run the application
echo "Starting Vision AI Desktop Application..."
$PYTHON_CMD main.py

# Return code
exit $?
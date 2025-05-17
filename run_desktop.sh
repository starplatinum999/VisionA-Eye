#!/bin/bash

# Vision AI Desktop App Launch Script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required dependencies are installed
echo "Checking dependencies..."

# Check PySide6
pip show PySide6 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "PySide6 not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting Vision AI Desktop Application..."
python main.py

# Return code
exit $? 
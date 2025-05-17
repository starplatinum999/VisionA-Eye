#!/bin/bash
# Launch script for Vision AI

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source visionai_fresh_env/Scripts/activate
else
    source visionai_fresh_env/bin/activate
fi

# Run the application with the fixed Python path
PYTHONPATH=$(pwd) streamlit run app.py

#!/bin/bash
# Vision AI launcher with PYTHONPATH fix

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run the app
streamlit run app.py

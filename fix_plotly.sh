#!/bin/bash
# Quick fix script for Plotly import issue

echo "Installing plotly package..."
pip install plotly==5.18.0

# Check if plotly can be imported now
python -c "import plotly.express" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Plotly installed successfully!"
    echo "You can now run: streamlit run app.py"
else
    echo "❌ Plotly installation failed."
    echo "Please run the full fix script: ./fix_environment.sh"
fi 
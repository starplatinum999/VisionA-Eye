#!/bin/bash
# Quick fix for Vision AI issues

echo "===================================================="
echo "Vision AI - Quick Fix"
echo "===================================================="

# Fix the test_installation.py issue with Python version check
echo "Fixing Python version check in test_installation.py..."
sed -i.bak 's/if py_version >= "3.8":/if int(py_version.split(".")[0]) >= 3 and int(py_version.split(".")[1]) >= 8 or int(py_version.split(".")[0]) > 3:/' test_installation.py

# Fix opencv-cv2 check to look for cv2 instead
echo "Fixing OpenCV check in test_installation.py..."
sed -i.bak 's/"opencv-cv2"/"cv2"/' test_installation.py

# Create a wrapper script that sets PYTHONPATH
echo "Creating a launcher script that sets PYTHONPATH..."
cat > run_vision_ai.sh << EOL
#!/bin/bash
# Vision AI launcher with PYTHONPATH fix

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH=\$(pwd):\$PYTHONPATH

# Run the app
streamlit run app.py
EOL

chmod +x run_vision_ai.sh

echo ""
echo "===================================================="
echo "Quick Fix Complete!"
echo "===================================================="
echo ""
echo "To run Vision AI, use the following command:"
echo ""
echo "./run_vision_ai.sh"
echo ""
echo "This will set the correct PYTHONPATH and run the application."
echo "" 
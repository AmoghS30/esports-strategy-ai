#!/bin/bash

echo "============================================================"
echo "    Installing Training Dependencies"
echo "============================================================"

# Check Python
python3 --version

# Create venv if needed
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Activate
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Install
echo "Installing packages..."
pip install --upgrade pip
pip install -r requirements_training.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To start training, run:"
echo "   ./run_training.sh"
echo ""

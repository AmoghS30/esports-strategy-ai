#!/bin/bash

# ESports Strategy AI - Installation Script

echo "============================================================"
echo "    ESports Strategy AI - Installation"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📦 Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "    Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GROQ_API_KEY"
echo "   Get your free key at: https://console.groq.com/"
echo ""
echo "2. Run the server:"
echo "   ./run.sh"
echo ""
echo "3. Open http://localhost:8000/docs in your browser"
echo ""

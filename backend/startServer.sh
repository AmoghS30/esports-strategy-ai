#!/bin/bash

# ESports Strategy Summarizer - Startup Script

echo "=========================================="
echo "ESports Strategy Summarizer - Backend"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check if model exists
if [ ! -d "models/esports-gpu-fast_final" ]; then
    echo ""
    echo "⚠️  Warning: Model directory not found!"
    echo "   Expected: models/esports-gpu-fast_final"
    echo "   Please ensure your trained model is available before starting."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo ""
    echo "📋 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo ""
    echo "ℹ️  No .env file found, using defaults"
    echo "   (Copy .env.example to .env to customize)"
fi

# Start the server
echo ""
echo "=========================================="
echo "🚀 Starting Flask Server"
echo "=========================================="
echo ""

# Check if running in production mode
if [ "$FLASK_ENV" = "production" ]; then
    echo "🔒 Running in PRODUCTION mode"
    echo ""
    
    # Check if gunicorn is available
    if ! command -v gunicorn &> /dev/null; then
        echo "⚠️  Gunicorn not found, falling back to Flask dev server"
        python3 esports_backend.py
    else
        echo "✅ Using Gunicorn for production"
        gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 esports_backend:app
    fi
else
    echo "🔧 Running in DEVELOPMENT mode"
    echo ""
    python3 esports_backend.py
fi
#!/bin/bash

# ESports Strategy AI - Run Script

echo "============================================================"
echo "    ESports Strategy AI - Starting Server"
echo "============================================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Copy .env.example to .env and add your GROQ_API_KEY"
    exit 1
fi

# Check if GROQ_API_KEY is set
if ! grep -q "GROQ_API_KEY=gsk_" .env; then
    echo "⚠️  Warning: GROQ_API_KEY may not be set correctly in .env"
    echo "   Get your free key at: https://console.groq.com/"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server
echo "🚀 Starting FastAPI server..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

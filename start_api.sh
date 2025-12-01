#!/bin/bash
# Medical AI Assistant API - Startup Script for Linux/Mac
# This script starts the FastAPI server

echo "========================================"
echo "Medical AI Assistant API"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Checking dependencies..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Set environment variables (use default values if not set)
if [ -z "$JWT_SECRET_KEY" ]; then
    echo "WARNING: JWT_SECRET_KEY not set, using default"
    export JWT_SECRET_KEY="default-secret-key-change-me"
fi

# Create necessary directories
mkdir -p logs
mkdir -p data/processed
mkdir -p data/raw
mkdir -p data/medical_literature

# Start the server
echo ""
echo "========================================"
echo "Starting Medical AI API Server"
echo "========================================"
echo ""
echo "Server will be available at:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo "  - Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Run with uvicorn
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Deactivate virtual environment on exit
deactivate

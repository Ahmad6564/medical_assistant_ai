@echo off
REM Medical AI Assistant API - Startup Script for Windows
REM This script starts the FastAPI server

echo ========================================
echo Medical AI Assistant API
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Checking dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Set environment variables (use default values if not set)
if not defined JWT_SECRET_KEY (
    echo WARNING: JWT_SECRET_KEY not set, using default
    set JWT_SECRET_KEY=default-secret-key-change-me
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data\processed" mkdir data\processed
if not exist "data\raw" mkdir data\raw
if not exist "data\medical_literature" mkdir data\medical_literature

REM Start the server
echo.
echo ========================================
echo Starting Medical AI API Server
echo ========================================
echo.
echo Server will be available at:
echo   - Swagger UI: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo   - Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Run with uvicorn
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

REM Deactivate virtual environment on exit
deactivate

pause

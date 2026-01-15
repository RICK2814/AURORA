@echo off
chcp 65001 >nul
echo ============================================================
echo    AURORA 2.0 - Mining Monitor
echo    Starting Streamlit Web Application...
echo ============================================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if app/main.py exists
if not exist "app\main.py" (
    echo ERROR: app\main.py not found!
    echo Current directory: %CD%
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo Starting Streamlit server...
echo.
echo The application will open in your default web browser.
echo If it doesn't open automatically, navigate to:
echo    http://localhost:8501
echo.
echo Press Ctrl+C to stop the server when you're done.
echo.
echo ============================================================
echo.

python -m streamlit run app\main.py

pause



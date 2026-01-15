@echo off
echo ============================================================
echo    Starting AURORA 2.0 Mining Monitor
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Checking Streamlit installation...
python -m streamlit --version
if errorlevel 1 (
    echo ERROR: Streamlit is not installed
    echo Installing Streamlit...
    pip install streamlit
)

echo.
echo ============================================================
echo Starting Streamlit server...
echo ============================================================
echo.
echo IMPORTANT: Keep this window open while using the app!
echo.
echo The app will be available at: http://localhost:8501
echo.
echo If your browser doesn't open automatically:
echo   1. Wait 10-15 seconds for the server to start
echo   2. Open your browser manually
echo   3. Go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m streamlit run app\main.py --server.headless false

pause



@echo off
echo ========================================
echo Starting AURORA 2.0 Mining Monitor
echo ========================================
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python -m streamlit run app/main.py
pause

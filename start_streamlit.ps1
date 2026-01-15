# PowerShell script to start Streamlit app
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AURORA 2.0 - Mining Monitor" -ForegroundColor Cyan
Write-Host "Starting Streamlit Web Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Current directory: $PWD" -ForegroundColor Yellow
Write-Host ""

# Check if app exists
if (Test-Path "app\main.py") {
    Write-Host "[OK] app\main.py found" -ForegroundColor Green
} else {
    Write-Host "[ERROR] app\main.py not found!" -ForegroundColor Red
    Write-Host "Please ensure you're running this from the project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting Streamlit server..." -ForegroundColor Yellow
Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start Streamlit
python -m streamlit run app/main.py --server.port 8501


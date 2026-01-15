# PowerShell script to launch Streamlit app
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   AURORA 2.0 - Mining Monitor" -ForegroundColor Cyan
Write-Host "   Starting Streamlit Web Application..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get the directory where this script is located
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if app/main.py exists
if (-not (Test-Path "app\main.py")) {
    Write-Host "ERROR: app\main.py not found!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting Streamlit server..." -ForegroundColor Green
Write-Host ""
Write-Host "The application will open in your default web browser." -ForegroundColor Yellow
Write-Host "If it doesn't open automatically, navigate to:" -ForegroundColor Yellow
Write-Host "   http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server when you're done." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start Streamlit
python -m streamlit run app\main.py



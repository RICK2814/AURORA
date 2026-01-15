"""
Launcher script for Streamlit app with error handling
"""
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def main():
    print("=" * 60)
    print("AURORA 2.0 - Mining Monitor")
    print("Starting Streamlit Web Application...")
    print("=" * 60)
    
    app_path = Path(__file__).parent / "app" / "main.py"
    
    if not app_path.exists():
        print(f"ERROR: App file not found at {app_path}")
        sys.exit(1)
    
    print(f"\nLaunching app from: {app_path}")
    print("\nThe app will open in your default web browser.")
    print("If it doesn't open automatically, navigate to:")
    print("  http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\nERROR: Failed to start Streamlit: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()


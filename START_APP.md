# How to Start the Web Application

## Quick Start

### Option 1: Using the Launcher Script (Recommended)
```bash
python run_app.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run app/main.py
```

### Option 3: Using Python Module
```bash
python -m streamlit run app/main.py
```

## What to Expect

1. **Server Starts**: Streamlit will start a local web server
2. **Browser Opens**: Your default browser should automatically open to `http://localhost:8501`
3. **If Browser Doesn't Open**: Manually navigate to `http://localhost:8501`

## Troubleshooting

### If you see import errors:

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### If port 8501 is already in use:

Streamlit will automatically try the next available port (8502, 8503, etc.)
Check the terminal output for the actual URL.

### To stop the server:

Press `Ctrl+C` in the terminal where Streamlit is running.

## Application Features

Once running, you'll have access to:

- **Home**: Project overview and quick start guide
- **Data Acquisition**: Download Sentinel-2 imagery
- **Detection Pipeline**: Run excavation detection
- **Temporal Analysis**: View time-series profiles
- **Violation Detection**: Check no-go zone violations
- **Visualizations**: Interactive maps and charts
- **Alert Log**: View violation alerts

## System Requirements

- Python 3.8+
- Streamlit installed (`pip install streamlit`)
- All dependencies from `requirements.txt`

## Note

Some features may require additional dependencies. If you see warnings about missing modules, install them using:
```bash
pip install <module_name>
```


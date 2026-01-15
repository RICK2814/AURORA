# How to Run the Streamlit App

## Quick Start - Choose One Method:

### Method 1: Double-Click Batch File (Easiest for Windows)
1. Navigate to the project folder: `AURORA 2.0 Mining Monitor system`
2. Double-click `LAUNCH_APP.bat`
3. Wait for the browser to open automatically
4. If browser doesn't open, go to: **http://localhost:8501**

### Method 2: PowerShell Script
1. Right-click on `LAUNCH_APP.ps1`
2. Select "Run with PowerShell"
3. If you get an execution policy error, run this first:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
4. Then run the script again

### Method 3: Command Line
Open PowerShell or Command Prompt in the project folder and run:

```bash
python -m streamlit run app\main.py
```

Or:

```bash
streamlit run app\main.py
```

## What Should Happen:

1. **Terminal Output**: You should see:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. **Browser Opens**: Your default browser should automatically open to the app

3. **If Browser Doesn't Open**: Manually navigate to **http://localhost:8501**

## Troubleshooting:

### Error: "Module not found" or Import Errors
Install dependencies:
```bash
pip install -r requirements.txt
```

### Error: "Port already in use"
Another Streamlit instance might be running. Either:
- Stop the other instance (Ctrl+C in its terminal)
- Or Streamlit will automatically use port 8502, 8503, etc.
- Check the terminal output for the actual URL

### Error: "Connection Refused"
1. Make sure Streamlit is actually running (check terminal)
2. Check if Windows Firewall is blocking it
3. Try a different port:
   ```bash
   streamlit run app\main.py --server.port 8502
   ```

### Error: "File not found"
Make sure you're running the command from the project root directory:
```bash
cd "C:\Users\Rohit\Downloads\AURORA 2.0 Mining Monitor system"
```

## Verify It's Working:

Once the app loads, you should see:
- **Home page** with project overview
- **Sidebar** with navigation menu
- **Multiple pages**: Data Acquisition, Detection Pipeline, etc.

## To Stop the Server:

Press **Ctrl+C** in the terminal where Streamlit is running.

## Need Help?

Check the terminal output for any error messages. Common issues:
- Missing Python packages → Install with `pip install -r requirements.txt`
- Wrong directory → Make sure you're in the project root
- Port conflicts → Use a different port or stop other instances



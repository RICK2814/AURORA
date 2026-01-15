# Quick Start Guide

## Prerequisites

1. Python 3.8 or higher
2. Git
3. Copernicus/Sentinel Hub account (free registration at https://dataspace.copernicus.eu/)

## Installation Steps

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "AURORA 2.0 Mining Monitor system"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `.env` file in the project root:

```env
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password
```

Or copy the example:
```bash
cp config.json.example config.json
# Edit config.json with your credentials
```

### 3. Prepare Data

Place your boundary files in `data/boundaries/`:
- Mine boundary: `mine_boundary.shp` (or `.geojson`)
- No-go zones: `no_go_zones.shp` (or `.geojson`)

### 4. Run the Application

**Option A: Streamlit Web Interface (Recommended)**
```bash
streamlit run app/main.py
```

**Option B: Command Line Interface**
```bash
# Download Sentinel-2 data
python -m src.main download --aoi data/boundaries/mine_boundary.shp --start-date 2025-01-01 --end-date 2026-01-11

# Preprocess images
python -m src.main preprocess --input-dir data/raw --aoi data/boundaries/mine_boundary.shp

# Detect excavation
python -m src.main detect --input-dir data/processed --method unsupervised

# Analyze temporal profile
python -m src.main analyze --masks-dir data/results
```

**Option C: Jupyter Notebooks**
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_signature_analysis.ipynb
# 3. notebooks/03_detection_pipeline.ipynb
# 4. notebooks/04_results_visualization.ipynb
```

## Example Workflow

1. **Data Acquisition**
   - Use the Streamlit interface or command line to download Sentinel-2 imagery
   - Specify your AOI, date range, and cloud cover threshold

2. **Preprocessing**
   - Preprocess downloaded .SAFE files
   - Cloud masking and band stacking will be applied automatically

3. **Detection**
   - Choose detection method (unsupervised, change detection, or ML)
   - Run detection on preprocessed images
   - Results saved as binary masks

4. **Analysis**
   - Generate temporal profiles
   - Calculate excavation rates
   - Detect violations in no-go zones

5. **Visualization**
   - View interactive maps and charts
   - Export results and reports

## Troubleshooting

### Common Issues

**Import Errors**
- Make sure virtual environment is activated
- Install requirements: `pip install -r requirements.txt`

**Authentication Errors**
- Verify Copernicus credentials in `.env` or `config.json`
- Check internet connection

**File Not Found Errors**
- Ensure data directories exist: `data/raw`, `data/processed`, `data/boundaries`, `data/results`
- Check file paths in configuration

**Memory Errors**
- Process images in smaller batches
- Reduce image resolution if needed

## Next Steps

- Read the full README.md for detailed documentation
- Explore example notebooks in `notebooks/`
- Check `docs/` for architecture and methodology details
- Review test files in `tests/` for usage examples

## Support

For issues or questions:
- Check the competition Discord server
- Contact event leads (see README.md)
- Review code comments and docstrings


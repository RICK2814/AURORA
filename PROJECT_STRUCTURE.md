# Project Structure

```
AURORA 2.0 Mining Monitor system/
│
├── README.md                 # Main project documentation
├── QUICKSTART.md            # Quick start guide
├── PROJECT_STRUCTURE.md     # This file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup configuration
├── .gitignore              # Git ignore rules
├── config.json.example     # Configuration template
│
├── data/                   # Data directory
│   ├── raw/               # Raw Sentinel-2 .SAFE files
│   ├── processed/         # Preprocessed GeoTIFF images
│   ├── boundaries/        # Shapefiles/GeoJSON (mine boundaries, no-go zones)
│   └── results/           # Detection results and outputs
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # CLI entry point
│   ├── utils.py           # Utility functions
│   ├── data_acquisition.py    # Sentinel-2 download
│   ├── preprocessing.py       # Image preprocessing
│   ├── feature_extraction.py  # Spectral indices
│   ├── excavation_detection.py # Core detection algorithms
│   ├── temporal_analysis.py  # Time-series analysis
│   ├── violation_detection.py # No-go zone violations
│   └── visualization.py      # Plotting and mapping
│
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_signature_analysis.ipynb
│   ├── 03_detection_pipeline.ipynb
│   └── 04_results_visualization.ipynb
│
├── app/                   # Web application
│   ├── main.py           # Streamlit app
│   ├── components/       # UI components (future)
│   └── static/           # Static assets (future)
│
├── docs/                  # Documentation
│   ├── mid_term_report.pdf (to be added)
│   ├── final_report.pdf (to be added)
│   └── architecture.png (to be added)
│
└── tests/                 # Unit tests
    ├── test_preprocessing.py
    └── test_detection.py
```

## Module Descriptions

### Core Modules (`src/`)

- **data_acquisition.py**: Handles downloading Sentinel-2 Level-2A products from Copernicus API
- **preprocessing.py**: Cloud masking, band stacking, resampling, clipping to AOI
- **feature_extraction.py**: Calculates spectral indices (NDVI, BSI, SWIR, NBR, etc.)
- **excavation_detection.py**: Adaptive detection using unsupervised clustering, change detection, or ML
- **temporal_analysis.py**: Time-series profiling, rate calculation, anomaly detection
- **violation_detection.py**: Detects and tracks violations in no-go zones, generates alerts
- **visualization.py**: Creates plots, maps, and interactive visualizations
- **utils.py**: Helper functions for file I/O, area calculations, alert logging

### Application (`app/`)

- **main.py**: Streamlit web interface with pages for:
  - Data acquisition
  - Detection pipeline
  - Temporal analysis
  - Violation detection
  - Visualizations
  - Alert log

### Notebooks (`notebooks/`)

- **01_data_exploration.ipynb**: Explore Sentinel-2 data and boundaries
- **02_signature_analysis.ipynb**: Analyze spectral signatures
- **03_detection_pipeline.ipynb**: Run complete detection pipeline
- **04_results_visualization.ipynb**: Visualize results and temporal profiles

## Data Flow

1. **Data Acquisition**: Download Sentinel-2 products → `data/raw/`
2. **Preprocessing**: Process .SAFE files → `data/processed/`
3. **Detection**: Detect excavation → `data/results/`
4. **Analysis**: Generate temporal profiles and detect violations
5. **Visualization**: Create maps, plots, and reports

## Key Files

- **requirements.txt**: All Python dependencies
- **config.json.example**: Configuration template
- **README.md**: Comprehensive project documentation
- **QUICKSTART.md**: Step-by-step setup guide


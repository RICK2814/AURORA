# AURORA 2.0 - Adaptive Mining Activity Monitor

An intelligent system that automatically detects, quantifies, and monitors excavation activity over time within legally defined mining areas and identifies violations in no-go zones using Sentinel-2 imagery.

## 🎯 Project Overview

This project addresses the challenge of building a **mine-agnostic, data-adaptive excavation monitoring system** that learns excavation signatures from data and reliably distinguishes mining from other land changes.

## 🏆 Competition Details

- **Competition**: AURORA 2.0 - PARSEC 6.0
- **Organizer**: IIT Dharwad
- **Knowledge Partner**: VEDAS, SAC-ISRO
- **Prize Pool**: ₹25,000

## 📋 Key Features

- **Adaptive Signature Learning**: Automatically infers spectral-temporal signatures of excavated surfaces without hard-coded thresholds
- **Temporal Profiling**: Generates time-series profiles of excavation activity with rate calculations
- **No-Go Violation Detection**: Detects and quantifies excavation in prohibited zones with automated alerts
- **Robust Handling**: Works across different mining commodities, land-cover types, and seasonal variations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Copernicus/Sentinel Hub account for data access
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "AURORA 2.0 Mining Monitor system"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with your credentials
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password
```

### Running the Application

**Streamlit Web Interface:**
```bash
streamlit run app/main.py
```

**Command Line Interface:**
```python
python src/main.py --mine_id <mine_id> --start_date 2025-01-01 --end_date 2026-01-11
```

## 📁 Project Structure

```
.
├── data/
│   ├── raw/              # Raw Sentinel-2 imagery
│   ├── processed/        # Processed and stacked images
│   ├── boundaries/       # Legal boundaries and no-go zones (shapefiles/GeoJSON)
│   └── results/          # Output results and visualizations
├── src/
│   ├── data_acquisition.py      # Sentinel-2 data download
│   ├── preprocessing.py         # Cloud masking, band stacking
│   ├── feature_extraction.py    # Spectral indices calculation
│   ├── excavation_detection.py  # Core detection algorithms
│   ├── temporal_analysis.py    # Time-series analysis
│   ├── violation_detection.py  # No-go zone violation detection
│   ├── visualization.py        # Plotting and mapping functions
│   └── utils.py                # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signature_analysis.ipynb
│   ├── 03_detection_pipeline.ipynb
│   └── 04_results_visualization.ipynb
├── app/
│   ├── main.py           # Streamlit main application
│   ├── components/       # UI components
│   └── static/          # Static assets
├── docs/
│   ├── mid_term_report.pdf
│   ├── final_report.pdf
│   └── architecture.png
├── tests/
│   ├── test_preprocessing.py
│   └── test_detection.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Core Pipeline

1. **Data Acquisition**: Download Sentinel-2 L2A imagery from Copernicus Browser
2. **Preprocessing**: Cloud masking, band stacking, clipping to AOI
3. **Feature Extraction**: Calculate spectral indices (NDVI, SWIR, BSI, NBR, etc.)
4. **Excavation Detection**: Adaptive clustering/change detection algorithms
5. **Temporal Analysis**: Generate time-series profiles and rate calculations
6. **Violation Detection**: Spatial intersection with no-go zones and alert generation
7. **Visualization**: Interactive maps, time-series plots, alert logs

## 📊 Expected Outputs

- **Time-Series Plots**: Excavated area over time, excavation rate, violation area
- **Spatial Maps**: Detected excavation regions with boundary overlays
- **Alert Log**: CSV/JSON with violation details (date, location, area, severity)

## 🛠️ Technical Stack

- **Python**: Geospatial processing (rasterio, geopandas, sentinelsat)
- **Machine Learning**: scikit-learn, tensorflow/pytorch
- **Visualization**: matplotlib, plotly, folium
- **Web Framework**: Streamlit (rapid prototyping)

## 📝 Usage Examples

### Basic Detection Pipeline

```python
from src.data_acquisition import download_sentinel2_data
from src.preprocessing import preprocess_images
from src.excavation_detection import detect_excavation
from src.temporal_analysis import analyze_temporal_profile

# Download data
products = download_sentinel2_data(
    aoi_polygon="data/boundaries/mine_boundary.shp",
    start_date="2025-01-01",
    end_date="2026-01-11"
)

# Preprocess
processed_images = preprocess_images(products)

# Detect excavation
excavation_masks = detect_excavation(processed_images)

# Analyze temporal profile
profile = analyze_temporal_profile(excavation_masks)
```

### Violation Detection

```python
from src.violation_detection import detect_violations

violations = detect_violations(
    excavation_masks=excavation_masks,
    no_go_zones="data/boundaries/no_go_zones.shp"
)

# Generate alerts
alerts = violations.generate_alerts()
```

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 📅 Timeline

- **Mid-term Submission**: January 4, 2026 EOD (20% weightage)
- **End-term Submission**: January 11, 2026 EOD (40% weightage)
- **Offline Presentation**: January 25, 2026 (40% weightage)

## 📧 Contact

- **Event Lead**: Aniruddh Pandav - ee23bt031@iitdh.ac.in
- **Event Co-Lead**: Balamurali V B - me24bt047@iitdh.ac.in
- **Events Lead PARSEC**: Dev Kaushal - outreach.parsec@iitdh.ac.in

## 📄 License

This project is developed for the AURORA 2.0 competition.

## 🙏 Acknowledgments

- IIT Dharwad for organizing the competition
- VEDAS, SAC-ISRO as knowledge partner
- Copernicus Programme for Sentinel-2 data


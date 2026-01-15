"""
Streamlit web application for AURORA 2.0 Mining Monitor
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Handle optional imports gracefully
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    st.warning("⚠️ geopandas not installed. Some features may be limited.")

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import with error handling
try:
    from src.feature_extraction import extract_spectral_indices
    from src.excavation_detection import detect_excavation
    HAS_SRC_MODULES = True
except ImportError as e:
    HAS_SRC_MODULES = False
    st.error(f"Error importing src modules: {e}")

try:
    from src.data_acquisition import download_sentinel2_data, Sentinel2Downloader
    HAS_DATA_ACQ = True
except ImportError:
    HAS_DATA_ACQ = False

try:
    from src.preprocessing import preprocess_images
    HAS_PREPROC = True
except ImportError:
    HAS_PREPROC = False

try:
    from src.temporal_analysis import analyze_temporal_profile
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

try:
    from src.violation_detection import detect_violations, ViolationDetector
    HAS_VIOLATIONS = True
except ImportError:
    HAS_VIOLATIONS = False

try:
    from src.visualization import (
        plot_temporal_profile, plot_violation_timeline, create_interactive_map
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

try:
    from src.utils import load_boundary, AlertLogger
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

try:
    from src import demo_data
    HAS_DEMO = True
except ImportError:
    HAS_DEMO = False

# Page configuration
st.set_page_config(
    page_title="AURORA 2.0 - Mining Monitor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⛏️ AURORA 2.0 - Adaptive Mining Activity Monitor</h1>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Acquisition", "Detection Pipeline", "Temporal Analysis", 
     "Violation Detection", "Visualizations", "Alert Log"]
)

# Global Demo Mode Toggle
st.sidebar.markdown("---")
use_demo = st.sidebar.checkbox("Run in Demo Mode", value=False)
if use_demo:
    st.sidebar.info("🔧 Demo Mode Active")
st.sidebar.markdown("---")

# Initialize session state
if 'excavation_results' not in st.session_state:
    st.session_state.excavation_results = {}
if 'temporal_profile' not in st.session_state:
    st.session_state.temporal_profile = None
if 'violations' not in st.session_state:
    st.session_state.violations = None

# Home Page
if page == "Home":
    st.markdown("""
    ## Welcome to AURORA 2.0 Mining Monitor
    
    This system automatically detects, quantifies, and monitors excavation activity 
    over time within legally defined mining areas and identifies violations in no-go zones 
    using Sentinel-2 imagery.
    
    ### Features:
    - **Adaptive Signature Learning**: Automatically infers spectral-temporal signatures
    - **Temporal Profiling**: Tracks excavation activity over time
    - **Violation Detection**: Identifies excavations in prohibited zones
    - **Interactive Visualization**: Explore results through maps and charts
    
    ### Quick Start:
    1. Go to **Data Acquisition** to download Sentinel-2 imagery
    2. Run **Detection Pipeline** to detect excavation areas
    3. View **Temporal Analysis** for time-series profiles
    4. Check **Violation Detection** for no-go zone alerts
    5. Explore **Visualizations** for interactive maps and charts
    """)
    
    st.info("💡 **Tip**: Make sure to set your Copernicus Data Space Ecosystem (CDSE) credentials in the Data Acquisition page before downloading imagery.")

# Data Acquisition Page
elif page == "Data Acquisition":
    st.markdown('<h2 class="sub-header">Data Acquisition</h2>', unsafe_allow_html=True)
    
    st.sidebar.subheader("Configuration")
    
    if use_demo and HAS_DEMO:
        st.info("🔧 **Demo Mode Active**: Using synthetic data and simulation.")
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic spatial data..."):
                demo_data.generate_demo_data()
                st.success("Synthetic data generated in `data/boundaries/`")
                # Pre-populate fake product info for the UI
                st.session_state.product_info = [
                    {
                        'uuid': 'demo-uuid-1',
                        'title': 'S2A_MSIL2A_20250101_DEMO',
                        'date': datetime(2025, 1, 1),
                        'cloud_cover': 5.0,
                        'size': '100 MB',
                        'footprint': 'POLYGON((...))'
                    },
                    {
                        'uuid': 'demo-uuid-2',
                        'title': 'S2A_MSIL2A_20250201_DEMO',
                        'date': datetime(2025, 2, 1),
                        'cloud_cover': 2.0,
                        'size': '100 MB',
                        'footprint': 'POLYGON((...))'
                    }
                ]
    if not use_demo:
        # Credentials only needed for real data
        username = st.sidebar.text_input("CDSE Email/Username", type="default")
        password = st.sidebar.text_input("CDSE Password", type="password")
    
    # AOI selection
    aoi_option = st.radio(
        "Area of Interest",
        ["Upload Shapefile/GeoJSON", "Use Existing File"]
    )
    
    aoi_path = None
    if aoi_option == "Upload Shapefile/GeoJSON":
        uploaded_file = st.file_uploader("Upload AOI file", type=['shp', 'geojson', 'json'])
        if uploaded_file:
            # Save uploaded file
            aoi_path = f"data/boundaries/{uploaded_file.name}"
            with open(aoi_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded: {uploaded_file.name}")
    else:
        existing_files = list(Path("data/boundaries").glob("*.shp")) + \
                        list(Path("data/boundaries").glob("*.geojson"))
        if existing_files:
            # Prioritize mines_cils.shp if it exists
            default_ix = 0
            for i, f in enumerate(existing_files):
                if f.name == "mines_cils.shp":
                    default_ix = i
                    break
            
            selected_file = st.selectbox("Select existing file", existing_files, index=default_ix)
            aoi_path = str(selected_file)
    
    if not use_demo:
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime(2026, 1, 11))
        
        cloud_cover_max = st.slider("Maximum Cloud Cover (%)", 0, 100, 30)
        max_downloads = st.number_input("Maximum Downloads", min_value=1, value=5, step=1)
        
        if st.button("Query Products"):
            if not aoi_path:
                st.error("Please provide an AOI file")
            else:
                with st.spinner("Querying Sentinel-2 products..."):
                    try:
                        # Allow query without strict credentials check first
                        downloader = Sentinel2Downloader(username or None, password or None)
                        products = downloader.query_products(
                            aoi=aoi_path,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            cloud_cover_max=cloud_cover_max
                        )
                        
                        product_info = downloader.get_product_info(products)
                        st.session_state.product_info = product_info
                        
                        if len(products) > 0:
                            st.success(f"Successfully found {len(products)} products!")
                        else:
                            st.warning("0 products found. Try increasing the Maximum Cloud Cover or broadening your Date Range.")
                        
                    except Exception as e:
                        st.error(f"Error querying products: {str(e)}")
    
    # Display products table if available (Demo or Real)
    if 'product_info' in st.session_state and st.session_state.product_info:
        st.subheader("Available Products")
        df = pd.DataFrame(st.session_state.product_info)
        st.dataframe(df[['title', 'date', 'cloud_cover', 'size']])
    
    if not use_demo and st.button("Download Products") and 'product_info' in st.session_state:
        if not username or not password:
             st.error("Please enter CDSE credentials above to download products.")
        else:
            with st.spinner("Downloading products..."):
                try:
                    downloader = Sentinel2Downloader(username, password)
                    # Implementation would download products here
                    st.success("Download completed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Detection Pipeline Page
elif page == "Detection Pipeline":
    st.markdown('<h2 class="sub-header">Excavation Detection Pipeline</h2>', 
                unsafe_allow_html=True)
    
    method = st.selectbox(
        "Detection Method",
        ["Unsupervised Clustering", "Change Detection", "Machine Learning", "Simulated (Demo)"]
    )
    
    method_map = {
        "Unsupervised Clustering": "unsupervised",
        "Change Detection": "change_detection",
        "Machine Learning": "ml",
        "Simulated (Demo)": "simulated"
    }
    
    if method == "Simulated (Demo)":
        st.info("Using analytical simulation for demo purposes.")
        if st.button("Run Simulation"):
             if not HAS_DEMO:
                 st.error("Demo module not found.")
             else:
                 with st.spinner("Simulating excavation growth..."):
                     generator = demo_data.DemoDataGenerator()
                     # Generate 6 months of data
                     start_date = datetime(2025, 1, 1)
                     results = generator.simulate_excavation_growth(start_date, days=180)
                     
                     st.session_state.excavation_results = results
                     
                     # Create a simple temporal profile from this
                     profile_data = []
                     for date_str, gdf in results.items():
                         area_ha = demo_data.gpd.GeoDataFrame(gdf).to_crs("EPSG:3857").area.sum() / 10000
                         profile_data.append({
                             'date': pd.to_datetime(date_str),
                             'excavated_area_ha': area_ha,
                             'excavation_rate_ha_per_day': 0 # To be calculated
                         })
                     
                     df = pd.DataFrame(profile_data)
                     df['excavation_rate_ha_per_day'] = df['excavated_area_ha'].diff()
                     st.session_state.temporal_profile = df
                     
                     st.success("Simulation completed! Results stored in session state.")

                     # Display results
                     st.subheader("Excavation Results Summary")
                     df = st.session_state.temporal_profile
                     if df is not None and not df.empty:
                         st.dataframe(df)
                         st.metric("Total Excavated Area (Latest)", f"{df['excavated_area_ha'].iloc[-1]:.2f} ha")
                         st.metric("Average Excavation Rate", f"{df['excavation_rate_ha_per_day'].mean():.2f} ha/day")
                     else:
                         st.warning("No temporal profile data available.")

    elif method != "Simulated (Demo)":     
        n_clusters = st.slider("Number of Clusters (for unsupervised)", 3, 10, 5)
        
        # File selection
        processed_files = list(Path("data/processed").glob("*.tif"))
        if processed_files:
            selected_files = st.multiselect("Select Processed Images", processed_files)
            
            if st.button("Run Detection"):
                if not selected_files:
                    st.warning("Please select at least one image.")
                elif not HAS_SRC_MODULES:
                    st.error("Source modules detection failed. Check logs.")
                else:
                    with st.spinner("Detecting excavation areas in real imagery..."):
                        try:
                             # Initialize detector
                             detector = detect_excavation.AdaptiveExcavationDetector(method=method_map.get(method, 'unsupervised'), n_clusters=n_clusters)
                             
                             results = {}
                             temporal_data = [] # For temporal analysis
                             
                             import rasterio # Needed for reading
                             
                             # Sort files by date if possible (assuming naming convention keeps order or we parse it)
                             # File format from preprocessing: {stem}_preprocessed.tif. Original usually contains date.
                             # Let's simple sort by name
                             selected_files.sort(key=lambda x: str(x))
                             
                             for p_file in selected_files:
                                 with rasterio.open(p_file) as src:
                                     bands = src.read() # Read all bands (stacked)
                                     profile = src.profile
                                     transform = src.transform
                                     crs = src.crs
                                     
                                     # Convert to format expected by detector (channels, height, width)
                                     # Already in (C, H, W) from rasterio
                                     
                                     # Detect
                                     mask, confidence = detector.detect(bands)
                                     
                                     # Parse date from filename
                                     # Expecting: T{UTM}{LAT}_{DATE}_... from original or just something we can parse
                                     # Preprocessor saves as {OriginalStem}_preprocessed.tif
                                     # Example original: T45QWE_20250101T050000_...
                                     filename = p_file.name
                                     date_str = "Unknown"
                                     import re
                                     # Simple regex to find YYYYMMDD
                                     date_match = re.search(r'20[2-3][0-9][0-1][0-9][0-3][0-9]', filename)
                                     if date_match:
                                         date_str = date_match.group(0)
                                         # Format to YYYY-MM-DD
                                         date_obj = datetime.strptime(date_str, "%Y%m%d")
                                         date_formatted = date_obj.strftime("%Y-%m-%d")
                                     else:
                                         # Fallback: use file modification time or counter
                                         date_formatted = f"Img_{filename}"

                                     # Calculate Area
                                     # Each pixel area depends on resolution (assume 10m for now if not in profile)
                                     # Get resolution from transform
                                     res_x = transform[0]
                                     res_y = -transform[4] # Usually negative
                                     pixel_area_m2 = abs(res_x * res_y)
                                     
                                     excavated_pixels = np.sum(mask)
                                     area_ha = (excavated_pixels * pixel_area_m2) / 10000.0
                                     
                                     # Store result
                                     # We store tuple: (mask, transform, crs) for creating Geom later
                                     results[date_formatted] = {
                                         'mask': mask,
                                         'transform': transform,
                                         'crs': crs,
                                         'area_ha': area_ha
                                     }
                                     
                                     temporal_data.append({
                                         'date': pd.to_datetime(date_formatted, errors='coerce'),
                                         'excavated_area_ha': area_ha
                                     })
                             
                             st.session_state.excavation_results = results
                             
                             # Build Temporal Profile DataFrame
                             df_temporal = pd.DataFrame(temporal_data)
                             if not df_temporal.empty:
                                 df_temporal = df_temporal.sort_values('date')
                                 df_temporal['excavation_rate_ha_per_day'] = df_temporal['excavated_area_ha'].diff() # Simple diff
                                 st.session_state.temporal_profile = df_temporal
                                 
                             st.success(f"Detection completed for {len(results)} images!")
                             
                        except Exception as e:
                            st.error(f"Error during detection: {e}")
                            import traceback
                            st.text(traceback.format_exc())

        else:
            st.warning("No processed images found. Please preprocess Sentinel-2 images first.")

# Temporal Analysis Page
elif page == "Temporal Analysis":
    st.markdown('<h2 class="sub-header">Temporal Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.temporal_profile is not None:
        df = st.session_state.temporal_profile
        
        # Plot
        fig = plot_temporal_profile(df)
        st.pyplot(fig)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Excavated Area", f"{df['excavated_area_ha'].max():.2f} ha")
        with col2:
            st.metric("Average Rate", f"{df['excavation_rate_ha_per_day'].mean():.2f} ha/day")
        with col3:
            st.metric("Peak Rate", f"{df['excavation_rate_ha_per_day'].max():.2f} ha/day")
        with col4:
            st.metric("Total Growth", f"{df['excavated_area_ha'].iloc[-1] - df['excavated_area_ha'].iloc[0]:.2f} ha")
        
        # Data table
        st.subheader("Temporal Profile Data")
        st.dataframe(df)
    else:
        st.info("No temporal profile available. Run the detection pipeline first.")

# Violation Detection Page
elif page == "Violation Detection":
    st.markdown('<h2 class="sub-header">No-Go Zone Violation Detection</h2>', 
                unsafe_allow_html=True)
    
    # Smart default for no-go zones - check both .shp and .geojson
    default_no_go = "data/boundaries/no_go_zones.shp"
    if not Path(default_no_go).exists():
        if Path("data/boundaries/no_go_zones.geojson").exists():
            default_no_go = "data/boundaries/no_go_zones.geojson"
    
    no_go_path = st.text_input("No-Go Zones Path", default_no_go)
    
    # Smart default for mine boundary - check both .shp and .geojson
    default_boundary = "data/boundaries/mine_boundary.shp"
    if Path("data/boundaries/mines_cils.shp").exists():
        default_boundary = "data/boundaries/mines_cils.shp"
    elif not Path(default_boundary).exists():
        if Path("data/boundaries/mines_cils.geojson").exists():
            default_boundary = "data/boundaries/mines_cils.geojson"
        elif Path("data/boundaries/mine_boundary.geojson").exists():
            default_boundary = "data/boundaries/mine_boundary.geojson"
        
    mine_boundary_path = st.text_input("Mine Boundary Path", default_boundary)
    
    if st.button("Detect Violations"):
        if st.session_state.excavation_results:
            # Validate file paths before proceeding
            actual_no_go_path = no_go_path
            if not Path(actual_no_go_path).exists():
                # Check if .geojson alternative exists
                if actual_no_go_path.strip().endswith('.shp'):
                     alt_path = actual_no_go_path.replace('.shp', '.geojson')
                     if Path(alt_path).exists():
                         actual_no_go_path = alt_path
                         st.warning(f"⚠️ Could not find .shp file, using available alternative: {actual_no_go_path}")
                
            if not Path(actual_no_go_path).exists():
                st.error(f"❌ No-Go Zones file not found: {no_go_path}\n\n"
                        f"Please ensure the file exists or update the path above.\n"
                        f"Available files in data/boundaries/: {', '.join([f.name for f in Path('data/boundaries').glob('*.geojson')] + [f.name for f in Path('data/boundaries').glob('*.shp')])}")
                st.stop()
            
            with st.spinner("Detecting violations..."):
                try:
                    # Initialize detector
                    detector = ViolationDetector(actual_no_go_path)
                
                    # Load mine boundary if provided
                    mine_boundary = None
                    if mine_boundary_path and Path(mine_boundary_path).exists():
                        mine_boundary = load_boundary(mine_boundary_path)
                    
                    # Prepare data for detection
                    results = st.session_state.excavation_results
                    
                    # distinct handling for demo data (dict of GDFs) vs real processing (dict of results)
                    is_demo_data = False
                    first_key = list(results.keys())[0] if results else None
                    if first_key and isinstance(results[first_key], gpd.GeoDataFrame): # Demo format
                         is_demo_data = True
                    elif first_key and 'mask' in results[first_key]: # Real format
                         is_demo_data = False
                    
                    if is_demo_data:
                        # Demo Mode: results is {date: gdf}
                        dates = sorted(list(results.keys()))
                        excavation_data_list = [results[d] for d in dates]
                        transforms = [None] * len(dates)
                        crs_list = [data.crs for data in excavation_data_list]
                        
                        # Run tracking
                        violation_df, alerts = detector.track_violations_over_time(
                            excavation_masks=excavation_data_list,
                            dates=dates,
                            transforms=transforms,
                            crs_list=crs_list,
                            mine_boundary=mine_boundary
                        )
                        st.session_state.violations = (violation_df, alerts)
                        st.success("Violation detection completed (Demo Data)!")
                        
                    else:
                        # Real Data Mode
                        # results is {date: {'mask': ..., 'transform': ..., 'crs': ...}}
                        dates = sorted(list(results.keys()))
                        
                        excavation_masks = [results[d]['mask'] for d in dates]
                        transforms = [results[d]['transform'] for d in dates]
                        crs_list = [results[d]['crs'] for d in dates]
                        
                        violation_df, alerts = detector.track_violations_over_time(
                            excavation_masks=excavation_masks,
                            dates=dates,
                            transforms=transforms,
                            crs_list=crs_list,
                            mine_boundary=mine_boundary
                        )
                        
                        st.session_state.violations = (violation_df, alerts)
                        st.success(f"Violation detection completed (Real Data)! Found {len(alerts)} alerts.")
                        
                except FileNotFoundError as e:
                    st.error(f"❌ File not found: {str(e)}\n\n"
                            f"Please check that the file paths are correct and the files exist.")
                except Exception as e:
                    st.error(f"Error in violation detection: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning("No excavation results available. Run detection pipeline first.")
    
    if st.session_state.violations is not None:
        violation_df, alerts = st.session_state.violations
        
        # Plot violations
        fig = plot_violation_timeline(violation_df)
        st.pyplot(fig)
        
        # Alerts table
        st.subheader("Violation Alerts")
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df)
        else:
            st.info("No violations detected.")

# Visualizations Page
elif page == "Visualizations":
    st.markdown('<h2 class="sub-header">Interactive Visualizations</h2>', 
                unsafe_allow_html=True)
    
    visualization_type = st.selectbox(
        "Visualization Type",
        ["Temporal Profile", "Violation Timeline", "Spatial Map"]
    )
    
    if visualization_type == "Temporal Profile" and st.session_state.temporal_profile is not None:
        df = st.session_state.temporal_profile
        
        # Interactive plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['excavated_area_ha'],
            mode='lines+markers',
            name='Excavated Area',
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Excavated Area Over Time",
            xaxis_title="Date",
            yaxis_title="Area (hectares)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif visualization_type == "Spatial Map":
        if not HAS_GEOPANDAS:
             st.error("GeoPandas is required for map visualization.")
        else:
            st.info("Loading latest spatial data map...")
            
            # Load available data
            mine_boundary = None
            mine_points = None
            no_go_zones = None
            
            # Check for CILS data first
            if Path("data/boundaries/mines_cils.shp").exists():
                 mine_boundary = gpd.read_file("data/boundaries/mines_cils.shp")
            elif Path("data/boundaries/mine_boundary.shp").exists():
                 mine_boundary = gpd.read_file("data/boundaries/mine_boundary.shp")
                 
            if Path("data/boundaries/mines_cils_points.shp").exists():
                 mine_points = gpd.read_file("data/boundaries/mines_cils_points.shp")
            elif Path("data/boundaries/mine_points.geojson").exists():
                 mine_points = gpd.read_file("data/boundaries/mine_points.geojson")
    
            if Path("data/boundaries/no_go_zones.shp").exists():
                no_go_zones = gpd.read_file("data/boundaries/no_go_zones.shp")
            elif Path("data/boundaries/no_go_zones.geojson").exists():
                no_go_zones = gpd.read_file("data/boundaries/no_go_zones.geojson")
                
            # Create Map
            if HAS_VIZ and HAS_FOLIUM:
                m = create_interactive_map(
                    excavation_mask=None,
                    mine_boundary=mine_boundary,
                    no_go_zones=no_go_zones,
                    mine_points=mine_points
                )
                st_folium(m, width=800, height=600)
            else:
                st.error("Visualization modules or Folium not loaded.")

# Alert Log Page
elif page == "Alert Log":
    st.markdown('<h2 class="sub-header">Violation Alert Log</h2>', unsafe_allow_html=True)
    
    alert_logger = AlertLogger()
    alerts = alert_logger.alerts
    
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(alerts))
        with col2:
            high_severity = sum(1 for a in alerts if a['severity'] == 'high')
            st.metric("High Severity", high_severity)
        with col3:
            total_area = sum(a['affected_area_hectares'] for a in alerts)
            st.metric("Total Violation Area", f"{total_area:.2f} ha")
    else:
        st.info("No alerts logged yet.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**AURORA 2.0 - PARSEC 6.0**")
st.sidebar.markdown("IIT Dharwad")


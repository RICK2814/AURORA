"""
Utility functions for the AURORA 2.0 Mining Monitor system
"""

import os
import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Union, Dict, List, Tuple
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict, config_path: str = "config.json"):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_boundary(file_path: str) -> gpd.GeoDataFrame:
    """Load boundary shapefile or GeoJSON, with automatic format detection"""
    file_path_obj = Path(file_path)
    
    # If file exists as specified, load it
    if file_path_obj.exists():
        return gpd.read_file(file_path)
    
    # Try alternative extensions if the specified file doesn't exist
    base_path = file_path_obj.with_suffix('')
    
    # Try .geojson if .shp was specified
    if file_path.endswith('.shp'):
        geojson_path = base_path.with_suffix('.geojson')
        if geojson_path.exists():
            return gpd.read_file(str(geojson_path))
    
    # Try .shp if .geojson was specified
    elif file_path.endswith('.geojson') or file_path.endswith('.json'):
        shp_path = base_path.with_suffix('.shp')
        if shp_path.exists():
            return gpd.read_file(str(shp_path))
    
    # If still not found, raise error with helpful message
    raise FileNotFoundError(
        f"Boundary file not found: {file_path}\n"
        f"Tried: {file_path}\n"
        f"Also tried alternative formats, but none were found."
    )


def get_aoi_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Get bounding box from GeoDataFrame (minx, miny, maxx, maxy)"""
    bounds = gdf.total_bounds
    return tuple(bounds)


def create_output_dir(base_path: str, subdir: str = None) -> Path:
    """Create output directory if it doesn't exist"""
    if subdir:
        output_dir = Path(base_path) / subdir
    else:
        output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calculate_area_hectares(gdf: gpd.GeoDataFrame, crs: str = "EPSG:4326") -> float:
    """Calculate area in hectares"""
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    # Convert to appropriate UTM zone for accurate area calculation
    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    utm_zone = int(np.floor((center_lon + 180) / 6) + 1)
    utm_crs = f"EPSG:{32600 + utm_zone}" if center_lon >= 0 else f"EPSG:{32700 + utm_zone}"
    
    gdf_utm = gdf.to_crs(utm_crs)
    area_m2 = gdf_utm.geometry.area.sum()
    area_ha = area_m2 / 10000
    return area_ha


def normalize_array(arr: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """Normalize array to [0, 1] range"""
    if min_val is None:
        min_val = np.nanmin(arr)
    if max_val is None:
        max_val = np.nanmax(arr)
    
    if max_val == min_val:
        return np.zeros_like(arr)
    
    normalized = (arr - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def apply_cloud_mask(image: np.ndarray, scl_band: np.ndarray) -> np.ndarray:
    """Apply cloud mask using Scene Classification Layer (SCL)"""
    # Valid pixels: 4 (vegetation), 5 (not vegetated), 6 (water), 7 (unclassified)
    # Remove: 0 (no data), 1 (saturated/defective), 2 (dark area), 3 (cloud shadow),
    #         8 (cloud medium), 9 (cloud high), 10 (thin cirrus), 11 (snow)
    valid_pixels = np.isin(scl_band, [4, 5, 6, 7])
    
    if len(image.shape) == 2:
        masked_image = image.copy()
        masked_image[~valid_pixels] = np.nan
    else:
        masked_image = image.copy()
        for i in range(image.shape[0]):
            masked_image[i, ~valid_pixels] = np.nan
    
    return masked_image


def resample_to_common_resolution(arrays: List[np.ndarray], target_resolution: int = 10) -> List[np.ndarray]:
    """Resample arrays to common resolution (placeholder - implement with rasterio)"""
    # This is a placeholder - actual implementation would use rasterio
    return arrays


def format_date(date_str: Union[str, datetime]) -> str:
    """Format date string to YYYYMMDD"""
    if isinstance(date_str, datetime):
        return date_str.strftime("%Y%m%d")
    return date_str


def parse_date_range(start_date: str, end_date: str) -> Tuple[str, str]:
    """Parse date range for Sentinel-2 query"""
    start = format_date(start_date)
    end = format_date(end_date)
    return (start, end)


def calculate_confidence_score(prediction: np.ndarray, features: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate confidence score for excavation predictions"""
    # Simple confidence based on feature consistency
    # Higher values indicate more confident predictions
    confidence = np.ones_like(prediction, dtype=float)
    
    # Reduce confidence in areas with high uncertainty
    if 'uncertainty' in features:
        confidence = 1.0 - features['uncertainty']
    
    return np.clip(confidence, 0, 1)


class AlertLogger:
    """Logger for violation alerts"""
    
    def __init__(self, output_path: str = "data/results/alerts.json"):
        self.output_path = output_path
        self.alerts = []
        self.load_existing()
    
    def load_existing(self):
        """Load existing alerts from file"""
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                self.alerts = json.load(f)
    
    def add_alert(self, date: str, location: Dict, area_ha: float, 
                  violation_type: str, severity: str, confidence: float):
        """Add a new alert"""
        alert = {
            "date": date,
            "location": location,
            "affected_area_hectares": area_ha,
            "violation_type": violation_type,  # "initial" or "expansion"
            "severity": severity,  # "low", "medium", "high"
            "confidence_score": confidence
        }
        self.alerts.append(alert)
        self.save()
    
    def save(self):
        """Save alerts to file"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.alerts, f, indent=4)
    
    def get_alerts_by_date(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get alerts filtered by date range"""
        filtered = self.alerts
        if start_date:
            filtered = [a for a in filtered if a['date'] >= start_date]
        if end_date:
            filtered = [a for a in filtered if a['date'] <= end_date]
        return filtered


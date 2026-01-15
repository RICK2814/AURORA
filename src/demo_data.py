"""
Synthetic data generation for AURORA 2.0 Demo Mode
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .utils import create_output_dir

class DemoDataGenerator:
    """Generates synthetic spatial data for demo purposes"""
    
    def __init__(self, output_dir: str = "data/boundaries"):
        self.output_dir = create_output_dir(output_dir)
        # Center coordinates (approximate location in India, mining region)
        self.center_lat = 21.0
        self.center_lon = 85.0
        self.size_deg = 0.05  # Approx 5km x 5km
        
    def generate_aoi(self) -> str:
        """Generate Area of Interest"""
        minx = self.center_lon - self.size_deg
        maxx = self.center_lon + self.size_deg
        miny = self.center_lat - self.size_deg
        maxy = self.center_lat + self.size_deg
        
        poly = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        
        output_path = self.output_dir / "aoi.geojson"
        gdf.to_file(output_path, driver="GeoJSON")
        return str(output_path)

    def generate_no_go_zones(self, num_zones: int = 3) -> str:
        """Generate random no-go zones within AOI"""
        zones = []
        
        # Generate some fixed zones relative to center
        # Zone 1: North-East (Forest)
        zones.append(box(
            self.center_lon + 0.01, self.center_lat + 0.01,
            self.center_lon + 0.04, self.center_lat + 0.04
        ))
        
        # Zone 2: South-West (River buffer)
        zones.append(box(
            self.center_lon - 0.04, self.center_lat - 0.04,
            self.center_lon - 0.01, self.center_lat - 0.01
        ))
        
        # Zone 3: North-West (Village)
        zones.append(box(
            self.center_lon - 0.03, self.center_lat + 0.02,
            self.center_lon - 0.02, self.center_lat + 0.03
        ))
        
        gdf = gpd.GeoDataFrame(
            {'id': range(len(zones)), 'type': ['Forest', 'River Buffer', 'Village']},
            geometry=zones, 
            crs="EPSG:4326"
        )
        
        output_path = self.output_dir / "no_go_zones.geojson"
        gdf.to_file(output_path, driver="GeoJSON")
        return str(output_path)

    def generate_mine_boundary(self) -> str:
        """Generate legal mining boundary"""
        # Central area
        poly = box(
            self.center_lon - 0.02, self.center_lat - 0.02,
            self.center_lon + 0.02, self.center_lat + 0.02
        )
        
        gdf = gpd.GeoDataFrame(
            {'id': [1], 'license': ['ML-2025-001']},
            geometry=[poly], 
            crs="EPSG:4326"
        )
        
        output_path = self.output_dir / "mine_boundary.geojson"
        gdf.to_file(output_path, driver="GeoJSON")
        return str(output_path)

    def generate_mine_points(self) -> str:
        """Generate point locations of mines"""
        # Get mine boundary to find centroids
        mine_boundary_path = self.output_dir / "mine_boundary.geojson"
        if not mine_boundary_path.exists():
            self.generate_mine_boundary()
            
        mine_boundary = gpd.read_file(mine_boundary_path)
        
        # Create points from centroids
        points = mine_boundary.copy()
        points['geometry'] = points.geometry.centroid
        
        output_path = self.output_dir / "mine_points.geojson"
        points.to_file(output_path, driver="GeoJSON")
        return str(output_path)

    def simulate_excavation_growth(self, start_date: datetime, days: int = 30) -> Dict:
        """
        Simulate excavation growth over time
        Returns a dictionary keyed by date string containing GeoDataFrames of excavation
        """
        results = {}
        
        # Start with a small excavation in the center (legal)
        current_poly = box(
            self.center_lon - 0.005, self.center_lat - 0.005,
            self.center_lon + 0.005, self.center_lat + 0.005
        )
        
        # Another excavation starting near a no-go zone (illegal)
        illegal_start = box(
            self.center_lon + 0.015, self.center_lat + 0.015,
            self.center_lon + 0.016, self.center_lat + 0.016
        )
        
        polys = [current_poly, illegal_start]
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Growth factor (expand slightly each day)
            new_polys = []
            for poly in polys:
                # Buffer to simulate expansion (approx 10m per day)
                # 0.0001 deg approx 11m
                expanded = poly.buffer(0.0001 * (i * 0.1 + 1)) 
                new_polys.append(expanded)
            
            polys = new_polys
            
            # Create mask-like result (using polygons for demo efficiency)
            # In a real raster pipeline, this would be a raster mask
            # For compatibility with our ViolationDetector, we'll return the polygons
            # The app will need to handle this 'demo' format
            
            gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
            results[date_str] = gdf
            
        return results

def generate_demo_data(output_dir: str = "data/boundaries"):
    """Convenience function to generate all demo data"""
    generator = DemoDataGenerator(output_dir)
    print("Generating AOI...")
    generator.generate_aoi()
    print("Generating No-Go Zones...")
    generator.generate_no_go_zones()
    print("Generating Mine Boundary...")
    generator.generate_mine_boundary()
    print("Generating Mine Points...")
    generator.generate_mine_points()
    print("Demo data generation complete.")

if __name__ == "__main__":
    generate_demo_data()

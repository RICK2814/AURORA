"""
Violation detection module for no-go zones
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .utils import load_boundary, calculate_area_hectares, AlertLogger


class ViolationDetector:
    """Detector for violations in no-go zones"""
    
    def __init__(self, no_go_zones_path: str, 
                 alert_thresholds: Dict = None):
        """
        Initialize violation detector
        
        Args:
            no_go_zones_path: Path to no-go zones shapefile/GeoJSON
            alert_thresholds: Dictionary of alert thresholds
        """
        self.no_go_zones = load_boundary(no_go_zones_path)
        
        # Default alert thresholds
        self.thresholds = alert_thresholds or {
            'initial_detection': 0.1,  # hectares
            'expansion_alert': 0.5,    # hectares
            'severity_high_area': 2.0,  # hectares
            'severity_high_rate': 1.0,  # hectares per month
            'severity_medium_area': 0.5  # hectares
        }
        
        self.alert_logger = AlertLogger()
        self.violation_history = []
    
    def mask_to_polygons(self, mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
        """
        Convert binary mask to polygons
        
        Args:
            mask: Binary mask
            transform: Rasterio transform
            crs: Coordinate reference system
        
        Returns:
            GeoDataFrame with polygons
        """
        from rasterio.features import shapes
        
        # Convert mask to polygons
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(mask.astype(np.uint8), transform=transform))
            if v == 1
        )
        
        geoms = []
        for result in results:
            geoms.append(result['geometry'])
        
        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
        
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        return gdf
    
    def detect_violations(self, excavation_data: Union[np.ndarray, gpd.GeoDataFrame],
                         transform, crs,
                         date: str,
                         mine_boundary: Optional[gpd.GeoDataFrame] = None) -> Dict:
        """
        Detect violations in no-go zones
        
        Args:
            excavation_data: Binary excavation mask (numpy array) or GeoDataFrame
            transform: Rasterio transform (needed if mask is array)
            crs: Coordinate reference system
            date: Date string
            mine_boundary: Optional mine boundary to exclude from violations
        
        Returns:
            Dictionary with violation information
        """
        # Convert mask to polygons if needed
        if isinstance(excavation_data, gpd.GeoDataFrame):
            excavation_gdf = excavation_data
        else:
            excavation_gdf = self.mask_to_polygons(excavation_data, transform, crs)
        
        if len(excavation_gdf) == 0:
            return {
                'date': date,
                'violations': [],
                'total_violation_area_ha': 0.0
            }
        
        # Ensure same CRS
        if excavation_gdf.crs != self.no_go_zones.crs:
            excavation_gdf = excavation_gdf.to_crs(self.no_go_zones.crs)
        
        # Find intersections with no-go zones
        violations = []
        total_violation_area = 0.0
        
        for idx, no_go_zone in self.no_go_zones.iterrows():
            # Find intersection
            intersection = excavation_gdf.intersection(no_go_zone.geometry)
            
            # Filter out empty geometries
            valid_intersections = intersection[~intersection.is_empty]
            
            if len(valid_intersections) > 0:
                # Calculate area
                violation_area_ha = calculate_area_hectares(
                    gpd.GeoDataFrame(geometry=valid_intersections, crs=self.no_go_zones.crs)
                )
                
                if violation_area_ha > 0:
                    total_violation_area += violation_area_ha
                    
                    violation_info = {
                        'no_go_zone_id': idx,
                        'date': date,
                        'area_ha': violation_area_ha,
                        'geometry': valid_intersections.unary_union
                    }
                    violations.append(violation_info)
        
        # Check if mine boundary is provided and exclude legal areas
        if mine_boundary is not None:
            # Remove violations that are within legal mine boundary
            if mine_boundary.crs != self.no_go_zones.crs:
                mine_boundary = mine_boundary.to_crs(self.no_go_zones.crs)
            
            filtered_violations = []
            for violation in violations:
                violation_gdf = gpd.GeoDataFrame(
                    geometry=[violation['geometry']],
                    crs=self.no_go_zones.crs
                )
                
                # Check if violation is outside mine boundary
                outside_mine = ~violation_gdf.within(mine_boundary.unary_union).any()
                
                if outside_mine:
                    filtered_violations.append(violation)
                else:
                    # Reduce violation area by intersection with mine boundary
                    intersection_with_mine = violation_gdf.intersection(
                        mine_boundary.unary_union
                    )
                    if not intersection_with_mine.is_empty.any():
                        violation_area_in_mine = calculate_area_hectares(
                            gpd.GeoDataFrame(geometry=intersection_with_mine, crs=self.no_go_zones.crs)
                        )
                        violation['area_ha'] -= violation_area_in_mine
                        if violation['area_ha'] > 0:
                            filtered_violations.append(violation)
            
            violations = filtered_violations
            total_violation_area = sum(v['area_ha'] for v in violations)
        
        return {
            'date': date,
            'violations': violations,
            'total_violation_area_ha': total_violation_area
        }
    
    def classify_severity(self, area_ha: float, growth_rate: float = None) -> str:
        """
        Classify violation severity
        
        Args:
            area_ha: Violation area in hectares
            growth_rate: Growth rate in hectares per month (optional)
        
        Returns:
            Severity level ('low', 'medium', 'high')
        """
        if area_ha >= self.thresholds['severity_high_area']:
            return 'high'
        
        if growth_rate and growth_rate >= self.thresholds['severity_high_rate']:
            return 'high'
        
        if area_ha >= self.thresholds['severity_medium_area']:
            return 'medium'
        
        return 'low'
    
    def generate_alerts(self, violation_results: List[Dict]) -> List[Dict]:
        """
        Generate alerts from violation results
        
        Args:
            violation_results: List of violation result dictionaries
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Track previous violations for expansion detection
        previous_violations = {}
        
        for result in violation_results:
            date = result['date']
            violations = result['violations']
            
            for violation in violations:
                zone_id = violation['no_go_zone_id']
                area_ha = violation['area_ha']
                
                # Check if this is initial detection or expansion
                if zone_id in previous_violations:
                    previous_area = previous_violations[zone_id]['area_ha']
                    area_growth = area_ha - previous_area
                    
                    if area_growth >= self.thresholds['expansion_alert']:
                        # Expansion alert
                        violation_type = 'expansion'
                        growth_rate = area_growth / 30  # Approximate monthly rate
                        severity = self.classify_severity(area_ha, growth_rate)
                        
                        alert = {
                            'date': date,
                            'location': {
                                'no_go_zone_id': zone_id,
                                'geometry': violation['geometry']
                            },
                            'area_ha': area_ha,
                            'violation_type': violation_type,
                            'severity': severity,
                            'area_growth_ha': area_growth,
                            'confidence_score': min(area_ha / 2.0, 1.0)  # Simple confidence
                        }
                        alerts.append(alert)
                        
                        # Log alert
                        self.alert_logger.add_alert(
                            date=date,
                            location={'no_go_zone_id': zone_id},
                            area_ha=area_ha,
                            violation_type=violation_type,
                            severity=severity,
                            confidence=alert['confidence_score']
                        )
                
                elif area_ha >= self.thresholds['initial_detection']:
                    # Initial detection alert
                    violation_type = 'initial'
                    severity = self.classify_severity(area_ha)
                    
                    alert = {
                        'date': date,
                        'location': {
                            'no_go_zone_id': zone_id,
                            'geometry': violation['geometry']
                        },
                        'area_ha': area_ha,
                        'violation_type': violation_type,
                        'severity': severity,
                        'confidence_score': min(area_ha / 2.0, 1.0)
                    }
                    alerts.append(alert)
                    
                    # Log alert
                    self.alert_logger.add_alert(
                        date=date,
                        location={'no_go_zone_id': zone_id},
                        area_ha=area_ha,
                        violation_type=violation_type,
                        severity=severity,
                        confidence=alert['confidence_score']
                    )
                
                # Update previous violations
                previous_violations[zone_id] = violation
        
        return alerts
    
    def track_violations_over_time(self, excavation_masks: List[np.ndarray],
                                  dates: List[str],
                                  transforms: List,
                                  crs_list: List,
                                  mine_boundary: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """
        Track violations over time
        
        Args:
            excavation_masks: List of excavation masks
            dates: List of date strings
            transforms: List of rasterio transforms
            crs_list: List of CRS strings
            mine_boundary: Optional mine boundary
        
        Returns:
            DataFrame with violation timeline
        """
        import pandas as pd
        
        violation_results = []
        
        for mask, date, transform, crs in zip(excavation_masks, dates, transforms, crs_list):
            result = self.detect_violations(mask, transform, crs, date, mine_boundary)
            violation_results.append(result)
        
        # Generate alerts
        alerts = self.generate_alerts(violation_results)
        
        # Create DataFrame
        df_data = []
        for result in violation_results:
            df_data.append({
                'date': result['date'],
                'total_violation_area_ha': result['total_violation_area_ha'],
                'n_violations': len(result['violations'])
            })
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df, alerts


def detect_violations(excavation_masks: List[np.ndarray],
                     dates: List[str],
                     no_go_zones_path: str,
                     transforms: List,
                     crs_list: List,
                     mine_boundary_path: Optional[str] = None) -> Tuple:
    """
    Convenience function for violation detection
    
    Args:
        excavation_masks: List of excavation masks
        dates: List of date strings
        no_go_zones_path: Path to no-go zones
        transforms: List of rasterio transforms
        crs_list: List of CRS strings
        mine_boundary_path: Optional path to mine boundary
    
    Returns:
        Tuple of (violation_dataframe, alerts_list)
    """
    detector = ViolationDetector(no_go_zones_path)
    
    mine_boundary = None
    if mine_boundary_path:
        mine_boundary = load_boundary(mine_boundary_path)
    
    return detector.track_violations_over_time(
        excavation_masks, dates, transforms, crs_list, mine_boundary
    )


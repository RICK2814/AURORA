"""
Temporal analysis module for tracking excavation over time
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.signal import savgol_filter
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

from .utils import calculate_area_hectares


class TemporalProfileAnalyzer:
    """Analyzer for temporal excavation profiles"""
    
    def __init__(self, pixel_size_m: float = 10.0):
        """
        Initialize analyzer
        
        Args:
            pixel_size_m: Pixel size in meters (default 10m for Sentinel-2)
        """
        self.pixel_size_m = pixel_size_m
        self.pixel_area_m2 = pixel_size_m ** 2
        self.pixel_area_ha = self.pixel_area_m2 / 10000
    
    def calculate_excavated_area(self, mask: np.ndarray) -> float:
        """
        Calculate excavated area in hectares
        
        Args:
            mask: Binary excavation mask
        
        Returns:
            Area in hectares
        """
        n_pixels = np.sum(mask)
        area_ha = n_pixels * self.pixel_area_ha
        return area_ha
    
    def calculate_excavation_rate(self, areas: List[float], dates: List[str]) -> pd.DataFrame:
        """
        Calculate excavation rate between consecutive dates
        
        Args:
            areas: List of excavated areas (hectares)
            dates: List of date strings
        
        Returns:
            DataFrame with dates, areas, and rates
        """
        df = pd.DataFrame({
            'date': dates,
            'excavated_area_ha': areas
        })
        
        # Convert dates to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate rate (area change per day)
        df['area_change'] = df['excavated_area_ha'].diff()
        df['days_diff'] = df['date'].diff().dt.days
        df['excavation_rate_ha_per_day'] = df['area_change'] / df['days_diff']
        
        # Calculate monthly rate (extrapolated)
        df['excavation_rate_ha_per_month'] = df['excavation_rate_ha_per_day'] * 30
        
        return df
    
    def smooth_temporal_series(self, values: np.ndarray, window_length: int = 5, 
                              polyorder: int = 2) -> np.ndarray:
        """
        Smooth temporal series using Savitzky-Golay filter
        
        Args:
            values: Time series values
            window_length: Window length (must be odd)
            polyorder: Polynomial order
        
        Returns:
            Smoothed values
        """
        if len(values) < window_length:
            return values
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window_length doesn't exceed data length
        window_length = min(window_length, len(values))
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length < 3:
            return values
        
        try:
            smoothed = savgol_filter(values, window_length, polyorder)
            return smoothed
        except:
            return values
    
    def detect_anomalies(self, areas: List[float], dates: List[str],
                       threshold_std: float = 2.0) -> List[Dict]:
        """
        Detect anomalies in temporal series
        
        Args:
            areas: List of excavated areas
            dates: List of date strings
            threshold_std: Standard deviation threshold for anomalies
        
        Returns:
            List of anomaly dictionaries
        """
        areas_array = np.array(areas)
        mean_area = np.nanmean(areas_array)
        std_area = np.nanstd(areas_array)
        
        anomalies = []
        for i, (date, area) in enumerate(zip(dates, areas)):
            z_score = abs((area - mean_area) / (std_area + 1e-10))
            if z_score > threshold_std:
                anomalies.append({
                    'date': date,
                    'area': area,
                    'z_score': z_score,
                    'type': 'high' if area > mean_area else 'low'
                })
        
        return anomalies
    
    def filter_seasonal_noise(self, areas: List[float], dates: List[str],
                             method: str = 'moving_average',
                             window: int = 3) -> List[float]:
        """
        Filter seasonal noise from temporal series
        
        Args:
            areas: List of excavated areas
            dates: List of date strings
            method: Filtering method ('moving_average' or 'savgol')
            window: Window size for filtering
        
        Returns:
            Filtered areas
        """
        areas_array = np.array(areas)
        
        if method == 'moving_average':
            # Simple moving average
            filtered = pd.Series(areas_array).rolling(window=window, center=True).mean().fillna(areas_array)
            return filtered.tolist()
        
        elif method == 'savgol':
            filtered = self.smooth_temporal_series(areas_array, window_length=window)
            return filtered.tolist()
        
        else:
            return areas
    
    def generate_temporal_profile(self, masks: List[np.ndarray],
                                 dates: List[str],
                                 smooth: bool = True,
                                 filter_seasonal: bool = True) -> pd.DataFrame:
        """
        Generate complete temporal profile
        
        Args:
            masks: List of excavation masks over time
            dates: List of date strings
            smooth: Whether to smooth the series
            filter_seasonal: Whether to filter seasonal noise
        
        Returns:
            DataFrame with temporal profile
        """
        # Calculate areas
        areas = [self.calculate_excavated_area(mask) for mask in masks]
        
        # Filter seasonal noise if requested
        if filter_seasonal:
            areas = self.filter_seasonal_noise(areas, dates)
        
        # Calculate rates
        df = self.calculate_excavation_rate(areas, dates)
        
        # Smooth if requested
        if smooth:
            df['excavated_area_ha_smoothed'] = self.smooth_temporal_series(
                df['excavated_area_ha'].values
            )
            df['excavation_rate_ha_per_day_smoothed'] = self.smooth_temporal_series(
                df['excavation_rate_ha_per_day'].fillna(0).values
            )
        
        # Detect anomalies
        anomalies = self.detect_anomalies(areas, dates)
        df['is_anomaly'] = df['date'].isin([a['date'] for a in anomalies])
        
        return df


def analyze_temporal_profile(masks: List[np.ndarray],
                            dates: List[str],
                            pixel_size_m: float = 10.0,
                            smooth: bool = True) -> pd.DataFrame:
    """
    Convenience function for temporal analysis
    
    Args:
        masks: List of excavation masks over time
        dates: List of date strings
        pixel_size_m: Pixel size in meters
        smooth: Whether to smooth the series
    
    Returns:
        DataFrame with temporal profile
    """
    analyzer = TemporalProfileAnalyzer(pixel_size_m=pixel_size_m)
    return analyzer.generate_temporal_profile(masks, dates, smooth=smooth)


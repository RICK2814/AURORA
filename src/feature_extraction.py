"""
Feature extraction module for calculating spectral indices
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SpectralIndices:
    """Class for calculating various spectral indices"""
    
    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            nir: Near-infrared band
            red: Red band
        
        Returns:
            NDVI array (values typically -1 to 1)
        """
        denominator = nir + red
        denominator[denominator == 0] = np.nan
        ndvi = (nir - red) / denominator
        return ndvi
    
    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index
        
        NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            green: Green band
            nir: Near-infrared band
        
        Returns:
            NDWI array
        """
        denominator = green + nir
        denominator[denominator == 0] = np.nan
        ndwi = (green - nir) / denominator
        return ndwi
    
    @staticmethod
    def calculate_bsi(swir1: np.ndarray, red: np.ndarray, 
                     nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Bare Soil Index
        
        BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        
        Args:
            swir1: SWIR1 band
            red: Red band
            nir: Near-infrared band
            blue: Blue band
        
        Returns:
            BSI array
        """
        numerator = (swir1 + red) - (nir + blue)
        denominator = (swir1 + red) + (nir + blue)
        denominator[denominator == 0] = np.nan
        bsi = numerator / denominator
        return bsi
    
    @staticmethod
    def calculate_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Burn Ratio
        
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        
        Args:
            nir: Near-infrared band
            swir2: SWIR2 band
        
        Returns:
            NBR array
        """
        denominator = nir + swir2
        denominator[denominator == 0] = np.nan
        nbr = (nir - swir2) / denominator
        return nbr
    
    @staticmethod
    def calculate_swir_ratio(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate SWIR ratio
        
        SWIR_ratio = SWIR1 / SWIR2
        
        Args:
            swir1: SWIR1 band
            swir2: SWIR2 band
        
        Returns:
            SWIR ratio array
        """
        swir2[swir2 == 0] = np.nan
        ratio = swir1 / swir2
        return ratio
    
    @staticmethod
    def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index
        
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Args:
            nir: Near-infrared band
            red: Red band
            blue: Blue band
        
        Returns:
            EVI array
        """
        denominator = nir + 6 * red - 7.5 * blue + 1
        denominator[denominator == 0] = np.nan
        evi = 2.5 * (nir - red) / denominator
        return evi


def extract_spectral_indices(bands: np.ndarray, 
                            band_order: list = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']) -> Dict[str, np.ndarray]:
    """
    Extract all spectral indices from band stack
    
    Args:
        bands: Stacked bands array (shape: [n_bands, height, width])
        band_order: List of band names in order
    
    Returns:
        Dictionary mapping index names to arrays
    """
    indices = SpectralIndices()
    
    # Map band names to indices
    band_map = {}
    for i, band_name in enumerate(band_order):
        band_map[band_name] = i
    
    # Extract bands
    blue = bands[band_map.get('B02', 0)] if 'B02' in band_map else None
    green = bands[band_map.get('B03', 1)] if 'B03' in band_map else None
    red = bands[band_map.get('B04', 2)] if 'B04' in band_map else None
    nir = bands[band_map.get('B08', 3)] if 'B08' in band_map else None
    swir1 = bands[band_map.get('B11', 4)] if 'B11' in band_map else None
    swir2 = bands[band_map.get('B12', 5)] if 'B12' in band_map else None
    
    spectral_indices = {}
    
    # Calculate indices
    if nir is not None and red is not None:
        spectral_indices['NDVI'] = indices.calculate_ndvi(nir, red)
    
    if green is not None and nir is not None:
        spectral_indices['NDWI'] = indices.calculate_ndwi(green, nir)
    
    if swir1 is not None and red is not None and nir is not None and blue is not None:
        spectral_indices['BSI'] = indices.calculate_bsi(swir1, red, nir, blue)
    
    if nir is not None and swir2 is not None:
        spectral_indices['NBR'] = indices.calculate_nbr(nir, swir2)
    
    if swir1 is not None and swir2 is not None:
        spectral_indices['SWIR_ratio'] = indices.calculate_swir_ratio(swir1, swir2)
    
    if nir is not None and red is not None and blue is not None:
        spectral_indices['EVI'] = indices.calculate_evi(nir, red, blue)
    
    return spectral_indices


def calculate_temporal_statistics(indices_series: Dict[str, list]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate temporal statistics for spectral indices
    
    Args:
        indices_series: Dictionary mapping index names to lists of arrays over time
    
    Returns:
        Dictionary with mean, std, min, max for each index
    """
    temporal_stats = {}
    
    for index_name, arrays in indices_series.items():
        if not arrays:
            continue
        
        # Stack arrays
        stacked = np.stack(arrays, axis=0)
        
        # Calculate statistics
        temporal_stats[index_name] = {
            'mean': np.nanmean(stacked, axis=0),
            'std': np.nanstd(stacked, axis=0),
            'min': np.nanmin(stacked, axis=0),
            'max': np.nanmax(stacked, axis=0),
            'median': np.nanmedian(stacked, axis=0)
        }
    
    return temporal_stats


def calculate_change_metrics(before: np.ndarray, after: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate change metrics between two time points
    
    Args:
        before: Array from earlier time
        after: Array from later time
    
    Returns:
        Dictionary with difference, percent change, etc.
    """
    change_metrics = {
        'difference': after - before,
        'percent_change': np.where(before != 0, 
                                  ((after - before) / before) * 100,
                                  np.nan),
        'absolute_change': np.abs(after - before)
    }
    
    return change_metrics


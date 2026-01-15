"""
Tests for preprocessing module
"""

import pytest
import numpy as np
from src.preprocessing import Sentinel2Preprocessor
from src.utils import apply_cloud_mask


def test_cloud_mask():
    """Test cloud masking functionality"""
    # Create dummy SCL band
    scl_band = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], size=(100, 100))
    
    # Create dummy image
    image = np.random.rand(100, 100) * 1000
    
    # Apply mask
    masked = apply_cloud_mask(image, scl_band)
    
    # Check that invalid pixels are NaN
    invalid_pixels = np.isin(scl_band, [0, 1, 2, 3, 8, 9, 10, 11])
    assert np.all(np.isnan(masked[invalid_pixels]))


def test_preprocessor_init():
    """Test preprocessor initialization"""
    preprocessor = Sentinel2Preprocessor(target_resolution=10)
    assert preprocessor.target_resolution == 10
    assert len(preprocessor.bands_10m) == 4
    assert len(preprocessor.bands_20m) == 6


"""
Tests for excavation detection module
"""

import pytest
import numpy as np
from src.excavation_detection import detect_excavation, AdaptiveExcavationDetector
from src.feature_extraction import extract_spectral_indices


def test_detector_init():
    """Test detector initialization"""
    detector = AdaptiveExcavationDetector(method='unsupervised', n_clusters=5)
    assert detector.method == 'unsupervised'
    assert detector.n_clusters == 5


def test_spectral_indices():
    """Test spectral index calculation"""
    # Create dummy bands (B02, B03, B04, B08, B11, B12)
    bands = np.random.rand(6, 100, 100) * 3000
    
    indices = extract_spectral_indices(bands)
    
    assert 'NDVI' in indices
    assert 'BSI' in indices or 'SWIR_ratio' in indices
    
    # Check NDVI range
    ndvi = indices['NDVI']
    assert np.nanmin(ndvi) >= -1
    assert np.nanmax(ndvi) <= 1


"""
Quick setup test script to validate all modules and dependencies
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    modules = [
        'src.utils',
        'src.data_acquisition',
        'src.preprocessing',
        'src.feature_extraction',
        'src.excavation_detection',
        'src.temporal_analysis',
        'src.violation_detection',
        'src.visualization',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")
    
    dependencies = [
        'numpy',
        'pandas',
        'geopandas',
        'rasterio',
        'matplotlib',
        'sklearn',
        'scipy',
        'folium',
        'streamlit',
    ]
    
    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  [OK] {dep}")
        except ImportError:
            print(f"  [FAIL] {dep} - not installed")
            failed.append(dep)
    
    return len(failed) == 0

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from src.feature_extraction import SpectralIndices
        
        # Test spectral index calculation
        nir = np.array([[1000, 2000], [1500, 1800]])
        red = np.array([[500, 800], [600, 700]])
        
        indices = SpectralIndices()
        ndvi = indices.calculate_ndvi(nir, red)
        
        assert ndvi.shape == nir.shape
        print("  [OK] Spectral index calculation works")
        
        # Test excavation detection
        from src.excavation_detection import AdaptiveExcavationDetector
        
        detector = AdaptiveExcavationDetector(method='unsupervised', n_clusters=3)
        assert detector.method == 'unsupervised'
        print("  [OK] Excavation detector initialization works")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Basic functionality test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/boundaries',
        'data/results',
        'src',
        'notebooks',
        'app',
        'tests',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [FAIL] {dir_path}/ - missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("AURORA 2.0 - Setup Validation Test")
    print("=" * 60)
    
    results = {
        'imports': test_imports(),
        'dependencies': test_dependencies(),
        'functionality': test_basic_functionality(),
        'directories': test_directory_structure(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "[OK]" if passed else "[FAIL]"
        print(f"{symbol} {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Setup is complete.")
    else:
        print("[WARNING] Some tests failed. Please install dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())


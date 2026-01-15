"""
Preprocessing module for Sentinel-2 imagery
Handles cloud masking, band stacking, and clipping
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

from .utils import load_boundary, apply_cloud_mask, create_output_dir


class Sentinel2Preprocessor:
    """Class for preprocessing Sentinel-2 Level-2A products"""
    
    def __init__(self, target_resolution: int = 10):
        """
        Initialize preprocessor
        
        Args:
            target_resolution: Target resolution in meters (10m or 20m)
        """
        self.target_resolution = target_resolution
        
        # Sentinel-2 band information (10m and 20m bands)
        self.bands_10m = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        self.bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # RedEdge, SWIR
        self.bands_60m = ['B01', 'B09', 'B10']  # Aerosol, Water vapor
        
        # SCL band for cloud masking
        self.scl_band = 'SCL'
    
    def load_safe_product(self, safe_path: str) -> Dict[str, str]:
        """
        Load paths to all bands in a SAFE product
        
        Args:
            safe_path: Path to .SAFE directory
        
        Returns:
            Dictionary mapping band names to file paths
        """
        safe_path = Path(safe_path)
        band_paths = {}
        
        # Find IMG_DATA directory
        img_data_dirs = list(safe_path.rglob('IMG_DATA'))
        if not img_data_dirs:
            raise ValueError(f"IMG_DATA directory not found in {safe_path}")
        
        img_data_dir = img_data_dirs[0]
        
        # Find resolution subdirectories
        for res_dir in ['R10m', 'R20m', 'R60m']:
            res_path = img_data_dir / res_dir
            if res_path.exists():
                for tif_file in res_path.glob('*.tif'):
                    # Extract band name from filename
                    # Format: T{UTM}{LAT}{DATE}_{BAND}_{RESOLUTION}.tif
                    parts = tif_file.stem.split('_')
                    if len(parts) >= 2:
                        band_name = parts[1]
                        band_paths[band_name] = str(tif_file)
        
        # Find SCL band (usually in different location)
        scl_files = list(safe_path.rglob('*SCL*.tif'))
        if scl_files:
            band_paths['SCL'] = str(scl_files[0])
        
        return band_paths
    
    def read_band(self, band_path: str, aoi: Optional[gpd.GeoDataFrame] = None) -> Tuple[np.ndarray, Dict]:
        """
        Read a single band and optionally clip to AOI
        
        Args:
            band_path: Path to band file
            aoi: Optional GeoDataFrame for clipping
        
        Returns:
            Tuple of (band_array, metadata_dict)
        """
        with rasterio.open(band_path) as src:
            if aoi is not None:
                # Clip to AOI
                shapes = [geom for geom in aoi.geometry]
                band_data, transform = mask(src, shapes, crop=True)
                band_data = band_data[0]  # Remove band dimension
                meta = src.meta.copy()
                meta.update({
                    'height': band_data.shape[0],
                    'width': band_data.shape[1],
                    'transform': transform
                })
            else:
                band_data = src.read(1)
                meta = src.meta.copy()
        
        return band_data, meta
    
    def resample_band(self, band_data: np.ndarray, src_meta: Dict, 
                     target_resolution: int, target_transform: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Resample band to target resolution
        
        Args:
            band_data: Input band array
            src_meta: Source metadata
            target_resolution: Target resolution in meters
            target_transform: Target transform (if resampling to match another band)
        
        Returns:
            Tuple of (resampled_array, new_metadata)
        """
        if target_transform is not None:
            # Resample to match target transform
            dst_shape = (target_transform['height'], target_transform['width'])
            dst_transform = target_transform['transform']
            dst_crs = target_transform['crs']
        else:
            # Calculate new transform based on target resolution
            src_transform = src_meta['transform']
            src_res = abs(src_transform[0])
            
            if src_res == target_resolution:
                return band_data, src_meta
            
            scale_factor = src_res / target_resolution
            dst_shape = (int(src_meta['height'] * scale_factor),
                        int(src_meta['width'] * scale_factor))
            
            # Calculate new transform
            dst_transform = src_transform * src_transform.scale(
                src_meta['width'] / dst_shape[1],
                src_meta['height'] / dst_shape[0]
            )
            dst_crs = src_meta['crs']
        
        # Create destination array
        dst_array = np.empty(dst_shape, dtype=src_meta['dtype'])
        
        # Reproject
        reproject(
            source=band_data,
            destination=dst_array,
            src_transform=src_meta['transform'],
            src_crs=src_meta['crs'],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        # Update metadata
        dst_meta = src_meta.copy()
        dst_meta.update({
            'height': dst_shape[0],
            'width': dst_shape[1],
            'transform': dst_transform
        })
        
        return dst_array, dst_meta
    
    def stack_bands(self, band_paths: Dict[str, str], 
                   aoi: Optional[gpd.GeoDataFrame] = None,
                   target_bands: List[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Stack multiple bands into a single array
        
        Args:
            band_paths: Dictionary mapping band names to file paths
            aoi: Optional GeoDataFrame for clipping
            target_bands: List of bands to include (None for all available)
        
        Returns:
            Tuple of (stacked_array, metadata_dict)
        """
        if target_bands is None:
            # Default bands: B02, B03, B04, B08, B11, B12
            target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        
        bands_data = []
        reference_meta = None
        
        # Read all bands
        for band_name in target_bands:
            if band_name not in band_paths:
                print(f"Warning: Band {band_name} not found, skipping...")
                continue
            
            band_data, meta = self.read_band(band_paths[band_name], aoi)
            
            if reference_meta is None:
                reference_meta = meta
                bands_data.append(band_data)
            else:
                # Resample to match reference resolution
                if meta['transform'] != reference_meta['transform']:
                    band_data, meta = self.resample_band(
                        band_data, meta, 
                        self.target_resolution,
                        {'height': reference_meta['height'],
                         'width': reference_meta['width'],
                         'transform': reference_meta['transform'],
                         'crs': reference_meta['crs']}
                    )
                bands_data.append(band_data)
        
        # Stack bands
        stacked = np.stack(bands_data, axis=0)
        
        # Update metadata
        stacked_meta = reference_meta.copy()
        stacked_meta.update({
            'count': len(bands_data),
            'dtype': stacked.dtype
        })
        
        return stacked, stacked_meta
    
    def preprocess_product(self, safe_path: str,
                          aoi: Optional[gpd.GeoDataFrame] = None,
                          apply_cloud_mask_flag: bool = True) -> Tuple[np.ndarray, Dict, Optional[np.ndarray]]:
        """
        Preprocess a complete Sentinel-2 product
        
        Args:
            safe_path: Path to .SAFE directory
            aoi: Optional GeoDataFrame for clipping
            apply_cloud_mask_flag: Whether to apply cloud mask
        
        Returns:
            Tuple of (stacked_bands, metadata, scl_band)
        """
        # Load band paths
        band_paths = self.load_safe_product(safe_path)
        
        # Stack bands
        stacked, meta = self.stack_bands(band_paths, aoi)
        
        # Load SCL band for cloud masking
        scl_band = None
        if apply_cloud_mask_flag and 'SCL' in band_paths:
            scl_data, scl_meta = self.read_band(band_paths['SCL'], aoi)
            if scl_meta['transform'] != meta['transform']:
                scl_data, scl_meta = self.resample_band(
                    scl_data, scl_meta, self.target_resolution,
                    {'height': meta['height'],
                     'width': meta['width'],
                     'transform': meta['transform'],
                     'crs': meta['crs']}
                )
            scl_band = scl_data
        
        # Apply cloud mask if requested
        if apply_cloud_mask_flag and scl_band is not None:
            for i in range(stacked.shape[0]):
                stacked[i] = apply_cloud_mask(stacked[i], scl_band)
        
        return stacked, meta, scl_band
    
    def save_preprocessed(self, stacked: np.ndarray, meta: Dict, 
                         output_path: str):
        """
        Save preprocessed image to GeoTIFF
        
        Args:
            stacked: Stacked band array
            meta: Metadata dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(stacked)


def preprocess_images(safe_paths: List[str],
                     aoi_path: Optional[str] = None,
                     output_dir: str = "data/processed",
                     apply_cloud_mask: bool = True) -> List[Dict]:
    """
    Preprocess multiple Sentinel-2 products
    
    Args:
        safe_paths: List of paths to .SAFE directories
        aoi_path: Optional path to AOI shapefile/GeoJSON
        output_dir: Output directory for preprocessed images
        apply_cloud_mask: Whether to apply cloud mask
    
    Returns:
        List of dictionaries with preprocessed data info
    """
    preprocessor = Sentinel2Preprocessor()
    
    # Load AOI if provided
    aoi = None
    if aoi_path and os.path.exists(aoi_path):
        aoi = load_boundary(aoi_path)
    
    # Create output directory
    output_dir = create_output_dir(output_dir)
    
    preprocessed_list = []
    
    for safe_path in safe_paths:
        try:
            print(f"Preprocessing {Path(safe_path).name}...")
            
            # Preprocess
            stacked, meta, scl = preprocessor.preprocess_product(
                safe_path, aoi, apply_cloud_mask
            )
            
            # Save preprocessed image
            output_filename = f"{Path(safe_path).stem}_preprocessed.tif"
            output_path = output_dir / output_filename
            preprocessor.save_preprocessed(stacked, meta, str(output_path))
            
            preprocessed_list.append({
                'original_path': safe_path,
                'preprocessed_path': str(output_path),
                'metadata': meta,
                'date': Path(safe_path).name.split('_')[2][:8]  # Extract date from filename
            })
            
        except Exception as e:
            print(f"Error preprocessing {safe_path}: {e}")
            continue
    
    return preprocessed_list


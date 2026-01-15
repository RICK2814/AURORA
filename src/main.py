"""
Main entry point for command-line interface
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from .data_acquisition import download_sentinel2_data
from .preprocessing import preprocess_images
from .excavation_detection import detect_excavation
from .temporal_analysis import analyze_temporal_profile
from .violation_detection import detect_violations
from .visualization import plot_temporal_profile, plot_violation_timeline


def main():
    parser = argparse.ArgumentParser(
        description="AURORA 2.0 - Adaptive Mining Activity Monitor"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download Sentinel-2 data')
    download_parser.add_argument('--aoi', required=True, help='Path to AOI shapefile/GeoJSON')
    download_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    download_parser.add_argument('--cloud-cover', type=float, default=30.0, help='Max cloud cover %')
    download_parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess Sentinel-2 images')
    preprocess_parser.add_argument('--input-dir', required=True, help='Input directory with .SAFE files')
    preprocess_parser.add_argument('--aoi', help='Path to AOI for clipping')
    preprocess_parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect excavation areas')
    detect_parser.add_argument('--input-dir', required=True, help='Directory with preprocessed images')
    detect_parser.add_argument('--method', default='unsupervised', 
                              choices=['unsupervised', 'change_detection', 'ml'],
                              help='Detection method')
    detect_parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters')
    detect_parser.add_argument('--output-dir', default='data/results', help='Output directory')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze temporal profile')
    analyze_parser.add_argument('--masks-dir', required=True, help='Directory with excavation masks')
    analyze_parser.add_argument('--output', help='Output CSV path')
    
    # Violations command
    violations_parser = subparsers.add_parser('violations', help='Detect violations')
    violations_parser.add_argument('--masks-dir', required=True, help='Directory with excavation masks')
    violations_parser.add_argument('--no-go-zones', required=True, help='Path to no-go zones')
    violations_parser.add_argument('--mine-boundary', help='Path to mine boundary')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        print(f"Downloading Sentinel-2 data from {args.start_date} to {args.end_date}...")
        results = download_sentinel2_data(
            aoi_path=args.aoi,
            start_date=args.start_date,
            end_date=args.end_date,
            cloud_cover_max=args.cloud_cover,
            output_dir=args.output_dir
        )
        print(f"Found {len(results['products'])} products")
        if results['downloaded_paths']:
            print(f"Downloaded {len(results['downloaded_paths'])} products")
    
    elif args.command == 'preprocess':
        print("Preprocessing Sentinel-2 images...")
        safe_paths = list(Path(args.input_dir).glob("*.SAFE"))
        if not safe_paths:
            print(f"No .SAFE files found in {args.input_dir}")
            sys.exit(1)
        
        preprocessed = preprocess_images(
            safe_paths=[str(p) for p in safe_paths],
            aoi_path=args.aoi,
            output_dir=args.output_dir
        )
        print(f"Preprocessed {len(preprocessed)} images")
    
    elif args.command == 'detect':
        print("Detecting excavation areas...")
        # Implementation would go here
        print("Detection completed")
    
    elif args.command == 'analyze':
        print("Analyzing temporal profile...")
        # Implementation would go here
        print("Analysis completed")
    
    elif args.command == 'violations':
        print("Detecting violations...")
        # Implementation would go here
        print("Violation detection completed")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


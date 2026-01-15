"""
Visualization module for maps, plots, and charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_temporal_profile(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot temporal profile of excavated area
    
    Args:
        df: DataFrame with temporal profile (from temporal_analysis)
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Excavated area over time
    ax1 = axes[0]
    ax1.plot(df['date'], df['excavated_area_ha'], 'o-', label='Raw', alpha=0.7)
    
    if 'excavated_area_ha_smoothed' in df.columns:
        ax1.plot(df['date'], df['excavated_area_ha_smoothed'], 
                '-', label='Smoothed', linewidth=2)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Excavated Area (hectares)', fontsize=12)
    ax1.set_title('Excavated Area Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Excavation rate
    ax2 = axes[1]
    if 'excavation_rate_ha_per_day' in df.columns:
        ax2.plot(df['date'], df['excavation_rate_ha_per_day'], 
                'o-', label='Rate (ha/day)', color='orange', alpha=0.7)
        
        if 'excavation_rate_ha_per_day_smoothed' in df.columns:
            ax2.plot(df['date'], df['excavation_rate_ha_per_day_smoothed'],
                    '-', label='Smoothed Rate', linewidth=2, color='darkorange')
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Excavation Rate (hectares/day)', fontsize=12)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Excavation Rate Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_violation_timeline(violation_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot violation timeline
    
    Args:
        violation_df: DataFrame with violation data
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(violation_df['date'], violation_df['total_violation_area_ha'],
           'o-', color='red', linewidth=2, markersize=8)
    ax.fill_between(violation_df['date'], violation_df['total_violation_area_ha'],
                    alpha=0.3, color='red')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Violation Area (hectares)', fontsize=12)
    ax.set_title('No-Go Zone Violations Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_map(excavation_mask: Optional[np.ndarray] = None,
                          transform = None,
                          crs = None,
                          date: Optional[str] = None,
                          mine_boundary: Optional[gpd.GeoDataFrame] = None,
                          no_go_zones: Optional[gpd.GeoDataFrame] = None,
                          mine_points: Optional[gpd.GeoDataFrame] = None,
                          center: Optional[Tuple[float, float]] = None) -> folium.Map:
    """
    Create interactive Folium map
    
    Args:
        excavation_mask: Optional binary excavation mask
        transform: Optional Rasterio transform
        crs: Optional Coordinate reference system
        date: Optional Date string
        mine_boundary: Optional mine boundary GeoDataFrame
        no_go_zones: Optional no-go zones GeoDataFrame
        mine_points: Optional mine points GeoDataFrame
        center: Optional map center (lat, lon)
    
    Returns:
        Folium map object
    """
    # Calculate center if not provided
    if center is None:
        if mine_boundary is not None:
            bounds = mine_boundary.total_bounds
            center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        elif no_go_zones is not None:
             bounds = no_go_zones.total_bounds
             center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        elif mine_points is not None:
             bounds = mine_points.total_bounds
             center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        elif transform is not None:
            # Use transform to get center
            center = (transform[5], transform[4])
        else:
            # Default center (approx India)
            center = (20.5937, 78.9629)
    
    # Create map
    m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')
    
    # Add mine boundary
    if mine_boundary is not None:
        if mine_boundary.crs != 'EPSG:4326':
            mine_boundary_4326 = mine_boundary.to_crs('EPSG:4326')
        else:
            mine_boundary_4326 = mine_boundary
        
        folium.GeoJson(
            mine_boundary_4326.to_json(),
            style_function=lambda feature: {
                'fillColor': 'green',
                'color': 'green',
                'weight': 2,
                'fillOpacity': 0.1
            },
            tooltip='Mine Boundary'
        ).add_to(m)
    
    # Add mine points
    if mine_points is not None:
        if mine_points.crs != 'EPSG:4326':
            mine_points_4326 = mine_points.to_crs('EPSG:4326')
        else:
            mine_points_4326 = mine_points
            
        # Add simpler CircleMarkers for points
        for idx, row in mine_points_4326.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=f"Mine Point {idx}"
            ).add_to(m)
    
    # Add no-go zones
    if no_go_zones is not None:
        if no_go_zones.crs != 'EPSG:4326':
            no_go_zones_4326 = no_go_zones.to_crs('EPSG:4326')
        else:
            no_go_zones_4326 = no_go_zones
        
        folium.GeoJson(
            no_go_zones_4326.to_json(),
            style_function=lambda feature: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.2
            },
            tooltip='No-Go Zone'
        ).add_to(m)
    
    # Add excavation mask if provided
    if excavation_mask is not None and transform is not None:
        pass # Placeholder for actual mask overlay logic (requires more complex specialized logic)
    
    # Add title
    title_text = f"Excavation Detection - {date}" if date else "Mining Monitor Map"
    title_html = f'''
    <h3 align="center" style="font-size:20px">
    <b>{title_text}</b>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def plot_spectral_indices(indices: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """
    Plot spectral indices as images
    
    Args:
        indices: Dictionary of spectral index arrays
        output_path: Optional path to save figure
    """
    n_indices = len(indices)
    n_cols = 3
    n_rows = (n_indices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_indices > 1 else [axes]
    
    for idx, (name, array) in enumerate(indices.items()):
        ax = axes[idx]
        im = ax.imshow(array, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for idx in range(n_indices, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_excavation_mask(mask: np.ndarray, rgb_image: Optional[np.ndarray] = None,
                        output_path: Optional[str] = None):
    """
    Plot excavation mask overlay
    
    Args:
        mask: Binary excavation mask
        rgb_image: Optional RGB image for background
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if rgb_image is not None:
        ax.imshow(rgb_image)
    
    # Overlay mask
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask == 1] = [1, 0, 0, 0.5]  # Red with transparency
    ax.imshow(mask_overlay)
    
    ax.set_title('Detected Excavation Areas', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Excavation')
    ax.legend(handles=[red_patch], loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(temporal_df: pd.DataFrame,
                            violation_df: Optional[pd.DataFrame] = None,
                            output_path: Optional[str] = None):
    """
    Create summary dashboard with multiple plots
    
    Args:
        temporal_df: Temporal profile DataFrame
        violation_df: Optional violation DataFrame
        output_path: Optional path to save figure
    """
    n_plots = 2 if violation_df is None else 3
    fig = plt.figure(figsize=(16, 5 * n_plots))
    
    # Plot 1: Excavated area
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(temporal_df['date'], temporal_df['excavated_area_ha'], 'o-', linewidth=2)
    ax1.set_title('Excavated Area Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Area (hectares)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Excavation rate
    ax2 = plt.subplot(n_plots, 1, 2)
    if 'excavation_rate_ha_per_day' in temporal_df.columns:
        ax2.plot(temporal_df['date'], temporal_df['excavation_rate_ha_per_day'], 
                'o-', color='orange', linewidth=2)
    ax2.set_title('Excavation Rate Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rate (hectares/day)')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Violations (if provided)
    if violation_df is not None:
        ax3 = plt.subplot(n_plots, 1, 3)
        ax3.plot(violation_df['date'], violation_df['total_violation_area_ha'],
                'o-', color='red', linewidth=2)
        ax3.set_title('No-Go Zone Violations Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Violation Area (hectares)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


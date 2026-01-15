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
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_temporal_profile(df: pd.DataFrame, output_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(df['date'], df['excavated_area_ha'], 'o-', label='Raw', alpha=0.7)

    if 'excavated_area_ha_smoothed' in df.columns:
        ax1.plot(
            df['date'],
            df['excavated_area_ha_smoothed'],
            '-',
            label='Smoothed',
            linewidth=2
        )

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Excavated Area (hectares)')
    ax1.set_title('Excavated Area Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    if 'excavation_rate_ha_per_day' in df.columns:
        ax2.plot(
            df['date'],
            df['excavation_rate_ha_per_day'],
            'o-',
            label='Rate (ha/day)',
            color='orange',
            alpha=0.7
        )

        if 'excavation_rate_ha_per_day_smoothed' in df.columns:
            ax2.plot(
                df['date'],
                df['excavation_rate_ha_per_day_smoothed'],
                '-',
                label='Smoothed Rate',
                linewidth=2,
                color='darkorange'
            )

        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Excavation Rate (hectares/day)')

    ax2.set_xlabel('Date')
    ax2.set_title('Excavation Rate Over Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_violation_timeline(violation_df: pd.DataFrame, output_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        violation_df['date'],
        violation_df['total_violation_area_ha'],
        'o-',
        color='red',
        linewidth=2
    )
    ax.fill_between(
        violation_df['date'],
        violation_df['total_violation_area_ha'],
        alpha=0.3,
        color='red'
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Violation Area (hectares)')
    ax.set_title('No-Go Zone Violations Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_interactive_map(
    excavation_mask: Optional[np.ndarray] = None,
    transform=None,
    crs=None,
    date: Optional[str] = None,
    mine_boundary: Optional[gpd.GeoDataFrame] = None,
    no_go_zones: Optional[gpd.GeoDataFrame] = None,
    mine_points: Optional[gpd.GeoDataFrame] = None,
    center: Optional[Tuple[float, float]] = None
) -> folium.Map:

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
        else:
            center = (20.5937, 78.9629)

    m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')

    # Add mine boundary
    if mine_boundary is not None:
        mine_boundary_4326 = (
            mine_boundary.to_crs('EPSG:4326')
            if mine_boundary.crs != 'EPSG:4326'
            else mine_boundary
        )

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

    # ✅ FIX #1 APPLIED HERE
    if mine_points is not None:
        mine_points_4326 = (
            mine_points.to_crs('EPSG:4326')
            if mine_points.crs != 'EPSG:4326'
            else mine_points
        )

        for idx, row in mine_points_4326.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=f"Mine Point {idx}"
            ).add_to(m)

    if no_go_zones is not None:
        no_go_zones_4326 = (
            no_go_zones.to_crs('EPSG:4326')
            if no_go_zones.crs != 'EPSG:4326'
            else no_go_zones
        )

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

    title_text = f"Excavation Detection - {date}" if date else "Mining Monitor Map"
    title_html = f"""
    <h3 align="center" style="font-size:20px">
        <b>{title_text}</b>
    </h3>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m

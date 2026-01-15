"""
Data acquisition module for downloading Sentinel-2 imagery
Updated to work with Copernicus Data Space Ecosystem (CDSE)
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import geopandas as gpd
from shapely import wkt
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from .utils import load_boundary, get_aoi_bounds, create_output_dir, format_date

load_dotenv()


class Sentinel2Downloader:
    """
    Class for downloading Sentinel-2 products from Copernicus Data Space Ecosystem
    Replaces legacy SciHub/sentinelsat implementation.
    """
    
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    search_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"

    def __init__(self, username: str = None, password: str = None):
        """
        Initialize Sentinel-2 downloader with CDSE credentials
        
        Args:
            username: Copernicus username (or from COPERNICUS_USERNAME env var)
            password: Copernicus password (or from COPERNICUS_PASSWORD env var)
        """
        self.username = username or os.getenv('COPERNICUS_USERNAME')
        self.password = password or os.getenv('COPERNICUS_PASSWORD')
        
        # Credentials are optional for searching, required for downloading
        self.session = requests.Session()
        self.authenticated = False
        
        if self.username and self.password:
            try:
                self._authenticate()
            except Exception as e:
                print(f"Warning: Authentication failed ({e}). Search may still work, but downloads will fail.")

    def _authenticate(self):
        """Authenticate with CDSE Identity Service (Keycloak)"""
        if not self.username or not self.password:
             self.authenticated = False
             return
             
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            response = self.session.post(self.auth_url, data=data)
            response.raise_for_status()
            token = response.json()["access_token"]
            # Only update headers if we actually got a token
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            self.authenticated = True
        except Exception as e:
            self.authenticated = False
            # Clear any failed auth headers
            if "Authorization" in self.session.headers:
                del self.session.headers["Authorization"]
            print(f"Authentication failed: {str(e)}")

    def query_products(self, 
                      aoi: str, 
                      start_date: str, 
                      end_date: str,
                      cloud_cover_max: float = 30.0,
                      product_type: str = 'S2MSI2A') -> Dict:
        """
        Query Sentinel-2 products via CDSE Resto API (Public)
        """
        # Ensure we have an absolute log path for debugging
        log_path = Path(os.getcwd()) / "debug_query.log"
        
        # Format dates
        def _fmt(d):
            if isinstance(d, datetime):
                return d.strftime("%Y-%m-%d")
            if isinstance(d, str):
                if len(d) == 8 and d.isdigit():
                    return f"{d[:4]}-{d[4:6]}-{d[6:]}"
                return d
            return d
            
        start = _fmt(start_date)
        end = _fmt(end_date)
        
        # Load AOI
        footprint = None
        # Handle path separators for Windows
        aoi_clean = aoi.replace('/', os.sep).replace('\\', os.sep)
        
        if os.path.exists(aoi_clean):
            if aoi_clean.endswith('.geojson') or aoi_clean.endswith('.json') or aoi_clean.endswith('.shp'):
                try:
                    gdf = gpd.read_file(aoi_clean)
                    if gdf.crs != "EPSG:4326":
                         gdf = gdf.to_crs("EPSG:4326")
                    # Use envelope (bounding box) for reliable searching
                    envelope = gdf.geometry.unary_union.envelope
                    footprint = envelope.wkt
                except Exception as e:
                    raise ValueError(f"Error reading AOI file: {e}")
            else:
                 with open(aoi_clean, 'r') as f:
                     footprint = f.read().strip()
        else:
            footprint = aoi
        
        # Prepare params
        params = {
            "startDate": start,
            "completionDate": end,
            "cloudCover": f"[0,{cloud_cover_max}]",
            "geometry": footprint,
            "productType": product_type,
            "maxRecords": 100,
            "sortParam": "startDate",
            "sortOrder": "descending"
        }
        
        try:
            # Debug log
            with open(log_path, "a") as f:
                f.write(f"\n--- {datetime.now()} ---\n")
                f.write(f"Querying for AOI: {aoi}\n")
                f.write(f"Footprint: {footprint}\n")
                f.write(f"Params: {json.dumps(params)}\n")

            # Search is typically public on CDSE Resto interface
            # If not authenticated, we use a clean request to avoid stale headers
            if not self.authenticated:
                response = requests.get(self.search_url, params=params)
            else:
                response = self.session.get(self.search_url, params=params)
            
            with open(log_path, "a") as f:
                f.write(f"Status: {response.status_code}\n")
                if response.status_code != 200:
                    f.write(f"Error Body: {response.text[:500]}\n")

            response.raise_for_status()
            features = response.json().get('features', [])
            
            with open(log_path, "a") as f:
                f.write(f"Found {len(features)} products in first attempt\n")
            
            # Fallback to S2MSI1C if S2MSI2A returns nothing
            if not features and product_type == 'S2MSI2A':
                params['productType'] = 'S2MSI1C'
                if not self.authenticated:
                    response = requests.get(self.search_url, params=params)
                else:
                    response = self.session.get(self.search_url, params=params)
                response.raise_for_status()
                features = response.json().get('features', [])
                with open(log_path, "a") as f:
                    f.write(f"Found {len(features)} products in fallback attempt\n")
            
            products = {}
            for feat in features:
                props = feat.get('properties', {})
                prod_id = feat.get('id')
                
                date_val = None
                if 'startDate' in props:
                    try:
                        date_val = datetime.strptime(props.get('startDate'), "%Y-%m-%dT%H:%M:%S.%fZ")
                    except ValueError:
                         try:
                             date_val = datetime.strptime(props.get('startDate'), "%Y-%m-%dT%H:%M:%SZ")
                         except ValueError:
                             pass

                products[prod_id] = {
                    'title': props.get('title'),
                    'beginposition': date_val,
                    'cloudcoverpercentage': float(props.get('cloudCover', 0)),
                    'size': 0,
                    'footprint': wkt.dumps(wkt.loads(feat.get('geometry'))) if isinstance(feat.get('geometry'), str) else json.dumps(feat.get('geometry')),
                    'download_url': props.get('services', {}).get('download', {}).get('url')
                }
            
            return products

        except Exception as e:
            print(f"Error querying products: {e}")
            return {}
    
    def download_product(self, product_id: str, output_dir: str = "data/raw", product_info: Optional[Dict] = None) -> str:
        """
        Download a single Sentinel-2 product (Requires Auth)
        """
        if not self.authenticated:
             # Try authenticating if credentials exist
             if self.username and self.password:
                 self._authenticate()
             else:
                 raise ValueError("Authentication required for download. Please provide valid CDSE credentials.")

        output_dir = create_output_dir(output_dir)
        
        download_url = None
        title = product_id
        
        if product_info and 'download_url' in product_info:
            download_url = product_info['download_url']
            title = product_info.get('title', product_id)
        else:
             download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

        filename = f"{title}.zip"
        output_path = output_dir / filename
        
        # Check if already extracted
        safe_name = f"{title}.SAFE"
        safe_path = output_dir / safe_name
        if safe_path.exists() and safe_path.is_dir():
            print(f"Product already exists: {safe_path}")
            return str(safe_path)
            
        if output_path.exists():
            print(f"Zip already exists, extracting...")
        else:
            print(f"Downloading {title}...")
            try:
                with self.session.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    # Check header for filename
                    if "Content-Disposition" in r.headers:
                        import re
                        fname_match = re.findall("filename=\"?([^\"]+)\"?", r.headers["Content-Disposition"])
                        if fname_match:
                            filename = fname_match[0]
                            output_path = output_dir / filename
                    
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                # Cleanup partial
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError(f"Download failed: {e}")

        # Unzip
        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Determine extracted path name
            # Usually strict match to title, but verify
            extracted_path = output_dir / output_path.stem
            if not extracted_path.exists():
                 # Try adding .SAFE
                 extracted_path = output_dir / f"{output_path.stem}.SAFE"
            
            if extracted_path.exists():
                return str(extracted_path)
            
            # Fallback scan
            possible = list(output_dir.glob(f"{output_path.stem}*.SAFE"))
            if possible:
                return str(possible[0])
                
            return str(output_path)
            
        except zipfile.BadZipFile:
            print("Error: Invalid zip file")
            return str(output_path)
        except Exception as e:
            print(f"Extraction failed: {e}")
            return str(output_path)

    def download_products(self, 
                         products: Dict, 
                         output_dir: str = "data/raw",
                         max_downloads: Optional[int] = None) -> List[str]:
        """
        Download multiple Sentinel-2 products
        """
        downloaded_paths = []
        product_list = list(products.items())
        
        if max_downloads:
            product_list = product_list[:max_downloads]
        
        for product_id, info in product_list:
            try:
                path = self.download_product(product_id, output_dir, info)
                downloaded_paths.append(path)
            except Exception as e:
                print(f"Error downloading {info.get('title', product_id)}: {e}")
                continue
        
        return downloaded_paths
    
    def get_product_info(self, products: Dict) -> List[Dict]:
        """
        Extract useful information from products dictionary
        """
        product_list = []
        for product_id, product_info in products.items():
            info = {
                'uuid': product_id,
                'title': product_info['title'],
                'date': product_info['beginposition'],
                'cloud_cover': product_info['cloudcoverpercentage'],
                'size': product_info.get('size', 'N/A'),
                'footprint': product_info['footprint'],
                'download_url': product_info.get('download_url')
            }
            product_list.append(info)
        
        # Sort by date
        product_list.sort(key=lambda x: x['date'] if x['date'] else datetime.min)
        return product_list


def download_sentinel2_data(aoi_path: str,
                            start_date: str,
                            end_date: str,
                            cloud_cover_max: float = 30.0,
                            output_dir: str = "data/raw",
                            download: bool = True,
                            max_downloads: Optional[int] = None) -> Dict:
    """
    Convenience function to download Sentinel-2 data
    """
    try:
        downloader = Sentinel2Downloader()
        
        # Query products
        print(f"Querying Sentinel-2 products from {start_date} to {end_date}...")
        products = downloader.query_products(
            aoi=aoi_path,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_max=cloud_cover_max
        )
        
        print(f"Found {len(products)} products")
        
        # Get product info
        product_info = downloader.get_product_info(products)
        
        result = {
            'products': products,
            'product_info': product_info,
            'downloaded_paths': []
        }
        
        # Download if requested
        if download and products:
            print("Downloading products...")
            downloaded_paths = downloader.download_products(
                products=products,
                output_dir=output_dir,
                max_downloads=max_downloads
            )
            result['downloaded_paths'] = downloaded_paths
        
        return result
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return {'error': str(e)}
    except Exception as e:
        print(f"Process Error: {e}")
        return {'error': str(e)}

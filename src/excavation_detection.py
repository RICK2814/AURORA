"""
Excavation detection module using adaptive learning approaches
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening
import warnings
warnings.filterwarnings('ignore')

from .feature_extraction import extract_spectral_indices


class AdaptiveExcavationDetector:
    """Adaptive excavation detector that learns signatures from data"""
    
    def __init__(self, method: str = 'unsupervised', n_clusters: int = 5):
        """
        Initialize detector
        
        Args:
            method: Detection method ('unsupervised', 'change_detection', 'ml')
            n_clusters: Number of clusters for unsupervised method
        """
        self.method = method
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = None
        self.excavation_cluster_id = None
    
    def detect_unsupervised(self, spectral_indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detect excavation using unsupervised clustering
        
        Args:
            spectral_indices: Dictionary of spectral indices
        
        Returns:
            Binary mask (1 = excavation, 0 = non-excavation)
        """
        # Select key indices for excavation detection
        key_indices = ['BSI', 'NDVI', 'SWIR_ratio', 'NBR']
        available_indices = [idx for idx in key_indices if idx in spectral_indices]
        
        if not available_indices:
            raise ValueError("No suitable spectral indices found")
        
        # Stack indices
        features_list = []
        for idx_name in available_indices:
            idx_array = spectral_indices[idx_name]
            # Flatten for clustering
            features_list.append(idx_array.flatten())
        
        features = np.column_stack(features_list)
        
        # Remove NaN values
        valid_mask = ~np.isnan(features).any(axis=1)
        features_valid = features[valid_mask]
        
        if len(features_valid) == 0:
            return np.zeros(spectral_indices[available_indices[0]].shape, dtype=bool)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features_valid)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Identify excavation cluster (high BSI, low NDVI)
        cluster_centers = kmeans.cluster_centers_
        
        # Find cluster with highest BSI and lowest NDVI (typical excavation signature)
        bsi_idx = available_indices.index('BSI') if 'BSI' in available_indices else 0
        ndvi_idx = available_indices.index('NDVI') if 'NDVI' in available_indices else 1
        
        # Score clusters: high BSI, low NDVI = excavation
        if 'BSI' in available_indices and 'NDVI' in available_indices:
            scores = cluster_centers[:, bsi_idx] - cluster_centers[:, ndvi_idx]
        else:
            # Use first available index
            scores = cluster_centers[:, 0]
        
        excavation_cluster = np.argmax(scores)
        self.excavation_cluster_id = excavation_cluster
        
        # Create mask
        mask_flat = np.zeros(features.shape[0], dtype=bool)
        mask_flat[valid_mask] = (labels == excavation_cluster)
        
        # Reshape to original dimensions
        mask = mask_flat.reshape(spectral_indices[available_indices[0]].shape)
        
        return mask
    
    def detect_change_detection(self, before_indices: Dict[str, np.ndarray],
                               after_indices: Dict[str, np.ndarray],
                               threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect excavation using change detection
        
        Args:
            before_indices: Spectral indices from earlier time
            after_indices: Spectral indices from later time
            threshold_percentile: Percentile for adaptive thresholding
        
        Returns:
            Binary mask (1 = excavation, 0 = non-excavation)
        """
        # Calculate change in key indices
        change_maps = {}
        
        if 'BSI' in before_indices and 'BSI' in after_indices:
            change_maps['BSI_change'] = after_indices['BSI'] - before_indices['BSI']
        
        if 'NDVI' in before_indices and 'NDVI' in after_indices:
            change_maps['NDVI_change'] = before_indices['NDVI'] - after_indices['NDVI']  # Negative change
        
        if 'SWIR_ratio' in before_indices and 'SWIR_ratio' in after_indices:
            change_maps['SWIR_change'] = after_indices['SWIR_ratio'] - before_indices['SWIR_ratio']
        
        if not change_maps:
            raise ValueError("No matching indices found for change detection")
        
        # Combine change maps (excavation: BSI increases, NDVI decreases, SWIR increases)
        combined_change = np.zeros_like(list(change_maps.values())[0])
        
        for change_name, change_array in change_maps.items():
            if 'BSI' in change_name or 'SWIR' in change_name:
                combined_change += np.nan_to_num(change_array, nan=0)
            elif 'NDVI' in change_name:
                combined_change += np.nan_to_num(change_array, nan=0)
        
        # Normalize
        combined_change = (combined_change - np.nanmin(combined_change)) / \
                         (np.nanmax(combined_change) - np.nanmin(combined_change) + 1e-10)
        
        # Adaptive thresholding
        threshold = np.nanpercentile(combined_change, threshold_percentile)
        mask = combined_change > threshold
        
        # Post-processing: morphological operations
        mask = binary_opening(mask, structure=np.ones((3, 3)))
        mask = binary_closing(mask, structure=np.ones((5, 5)))
        
        return mask
    
    def detect_ml(self, spectral_indices: Dict[str, np.ndarray],
                 training_samples: Optional[Dict] = None) -> np.ndarray:
        """
        Detect excavation using machine learning (requires training samples)
        
        Args:
            spectral_indices: Dictionary of spectral indices
            training_samples: Dictionary with 'excavated' and 'non_excavated' arrays
        
        Returns:
            Binary mask (1 = excavation, 0 = non-excavation)
        """
        if training_samples is None:
            # Fall back to unsupervised if no training data
            return self.detect_unsupervised(spectral_indices)
        
        # Extract features
        key_indices = ['BSI', 'NDVI', 'SWIR_ratio', 'NBR']
        available_indices = [idx for idx in key_indices if idx in spectral_indices]
        
        if not available_indices:
            raise ValueError("No suitable spectral indices found")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for label, samples in training_samples.items():
            for sample in samples:
                features = []
                for idx_name in available_indices:
                    if idx_name in spectral_indices:
                        features.append(spectral_indices[idx_name][sample[0], sample[1]])
                if len(features) == len(available_indices):
                    X_train.append(features)
                    y_train.append(1 if label == 'excavated' else 0)
        
        if not X_train:
            return self.detect_unsupervised(spectral_indices)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        # Predict on full image
        features_list = []
        for idx_name in available_indices:
            features_list.append(spectral_indices[idx_name].flatten())
        
        features = np.column_stack(features_list)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Reshape to original dimensions
        mask = predictions.reshape(spectral_indices[available_indices[0]].shape)
        confidence = probabilities.reshape(spectral_indices[available_indices[0]].shape)
        
        return mask, confidence
    
    def detect(self, bands: np.ndarray,
               before_bands: Optional[np.ndarray] = None,
               training_samples: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Main detection method
        
        Args:
            bands: Current band stack
            before_bands: Previous band stack (for change detection)
            training_samples: Training samples (for ML method)
        
        Returns:
            Tuple of (excavation_mask, confidence_map)
        """
        # Extract spectral indices
        spectral_indices = extract_spectral_indices(bands)
        
        if self.method == 'unsupervised':
            mask = self.detect_unsupervised(spectral_indices)
            return mask, None
        
        elif self.method == 'change_detection':
            if before_bands is None:
                # Fall back to unsupervised
                mask = self.detect_unsupervised(spectral_indices)
                return mask, None
            
            before_indices = extract_spectral_indices(before_bands)
            mask = self.detect_change_detection(before_indices, spectral_indices)
            return mask, None
        
        elif self.method == 'ml':
            result = self.detect_ml(spectral_indices, training_samples)
            if isinstance(result, tuple):
                return result
            else:
                return result, None
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


def detect_excavation(bands: np.ndarray,
                     method: str = 'unsupervised',
                     before_bands: Optional[np.ndarray] = None,
                     n_clusters: int = 5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function for excavation detection
    
    Args:
        bands: Current band stack
        method: Detection method ('unsupervised', 'change_detection', 'ml')
        before_bands: Previous band stack (for change detection)
        n_clusters: Number of clusters for unsupervised method
    
    Returns:
        Tuple of (excavation_mask, confidence_map)
    """
    detector = AdaptiveExcavationDetector(method=method, n_clusters=n_clusters)
    return detector.detect(bands, before_bands)


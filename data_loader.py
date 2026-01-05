"""
Data loader for UMAP clustering application.
Supports both Parquet/NPY mode and HDF5 mode.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import h5py


@dataclass
class FlatData:
    """Container for loaded vocalization data."""
    
    segments: pd.DataFrame  # Per-vocalization data
    files: pd.DataFrame  # Per-file data
    parameters: Dict[str, Any]  # Configuration parameters
    spectrograms_dir: Optional[Path] = None  # Directory containing spectrogram .npy files (Parquet mode)
    hdf5_path: Optional[Path] = None  # HDF5 file path (HDF5 mode)
    
    # Cached spectrograms (loaded on demand)
    _spectrogram_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _segment_spec_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    
    @property
    def n_segments(self) -> int:
        return len(self.segments)
    
    @property
    def n_files(self) -> int:
        return len(self.files)
    
    @property
    def scanrate(self) -> int:
        """Get processing sample rate (for spectrogram slicing)."""
        # Try new name first, then legacy name
        if 'sr_processing' in self.parameters:
            return self.parameters['sr_processing']
        if 'scanrate' in self.parameters:
            return self.parameters['scanrate']
        return 32000  # Default preprocessing rate
    
    @property
    def sr_original(self) -> Optional[int]:
        """Get original wav file sample rate (if stored)."""
        return self.parameters.get('sr_original')
    
    @property
    def umap_coords(self) -> np.ndarray:
        """Get UMAP coordinates as (N, 2) array."""
        # New format: 'umap' matrix (stored as list of arrays in DataFrame)
        if 'umap' in self.segments.columns:
            return np.stack(self.segments['umap'].values)
        # Legacy format: separate umap_x, umap_y columns
        if 'umap_x' in self.segments.columns:
            return self.segments[['umap_x', 'umap_y']].values
        return np.zeros((len(self.segments), 2))
    
    @property
    def pc_coords(self) -> Optional[np.ndarray]:
        """Get PC coordinates if available."""
        # New format: 'pca' matrix (stored as list of arrays in DataFrame)
        if 'pca' in self.segments.columns:
            return np.stack(self.segments['pca'].values)
        # Legacy format: separate pc_0, pc_1, ... columns
        pc_cols = [c for c in self.segments.columns if c.startswith('pc_')]
        if pc_cols:
            return self.segments[pc_cols].values
        return None
    
    def get_spectrogram(self, file_id: int) -> Optional[np.ndarray]:
        """Load spectrogram for a file (cached)."""
        if file_id in self._spectrogram_cache:
            return self._spectrogram_cache[file_id]
        
        if self.hdf5_path is not None:
            # HDF5 mode - try multiple key formats
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'spectrograms' not in f:
                    return None
                
                # Try different key formats
                possible_keys = [
                    f'spectrograms/file_{file_id:04d}',  # file_0000, file_0001, ...
                    f'spectrograms/{file_id}',           # 0, 1, 2, ...
                    f'spectrograms/{file_id:d}',         # Same as above
                ]
                
                for spec_key in possible_keys:
                    if spec_key in f:
                        spec = f[spec_key][:]
                        self._spectrogram_cache[file_id] = spec
                        return spec
                
                # Also check if key exists directly in spectrograms group
                spec_grp = f['spectrograms']
                for key in [f'file_{file_id:04d}', str(file_id)]:
                    if key in spec_grp:
                        spec = spec_grp[key][:]
                        self._spectrogram_cache[file_id] = spec
                        return spec
                        
        elif self.spectrograms_dir is not None:
            # Parquet mode
            spec_file = self.spectrograms_dir / f"{file_id}.npy"
            if spec_file.exists():
                spec = np.load(spec_file)
                self._spectrogram_cache[file_id] = spec
                return spec
        
        return None
    
    def get_segment_spectrogram(self, segment_id: int) -> Optional[np.ndarray]:
        """Extract spectrogram slice for a specific segment (cached)."""
        # Check cache first
        if segment_id in self._segment_spec_cache:
            return self._segment_spec_cache[segment_id]
        
        row = self.segments.iloc[segment_id]
        file_id = int(row['file_id'])
        onset_sec = float(row['onset_sec'])
        duration_sec = float(row['duration_sec'])
        
        # Compute samples from seconds using stored sample rate
        sr = self.scanrate
        onset_sample = int(onset_sec * sr)
        duration_samples = int(duration_sec * sr)
        
        full_spec = self.get_spectrogram(file_id)
        if full_spec is None:
            return None
        
        # Convert samples to spectrogram columns
        # hop_length = hop size in samples (e.g., 128)
        hop_length = self.parameters.get('hop_length', self.parameters.get('nonoverlap', 128))
        start_col = onset_sample // hop_length
        end_col = (onset_sample + duration_samples) // hop_length
        
        # Clamp to valid range
        start_col = max(0, start_col)
        end_col = min(full_spec.shape[1], end_col)
        
        segment_spec = full_spec[:, start_col:end_col]
        
        # Cache it
        self._segment_spec_cache[segment_id] = segment_spec
        return segment_spec
    
    def precache_all_spectrograms(self) -> None:
        """Pre-load all segment spectrograms into cache for faster access."""
        print(f"Pre-caching {self.n_segments} spectrograms...")
        for seg_id in range(self.n_segments):
            self.get_segment_spectrogram(seg_id)
            if (seg_id + 1) % 500 == 0:
                print(f"  Cached {seg_id + 1}/{self.n_segments}...")
        print(f"Done caching {self.n_segments} spectrograms.")
    
    def save_clusters(self, output_path: str) -> None:
        """Save current cluster assignments to file."""
        output_path = Path(output_path)
        
        # Save as parquet (full data)
        self.segments.to_parquet(output_path.with_suffix('.parquet'), index=False)
        
        # Also save as CSV for easy viewing
        export_df = self.segments[['segment_id', 'file_id', 'onset_sec', 'duration_sec', 'cluster_id']].copy()
        export_df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        print(f"Saved cluster assignments to {output_path}")
    
    def reset_clusters(self) -> None:
        """Reset all cluster assignments to 0."""
        self.segments['cluster_id'] = 0
        print("Reset all clusters to 0")


def load_data(path: str) -> FlatData:
    """Load data from path (auto-detect format)."""
    path = Path(path)
    
    if path.suffix == '.h5':
        return _load_from_hdf5(path)
    elif path.is_dir():
        return _load_from_parquet_dir(path)
    elif path.suffix == '.parquet':
        return _load_from_parquet_file(path)
    else:
        raise ValueError(f"Unknown data format: {path}")


def _load_from_hdf5(hdf5_path: Path) -> FlatData:
    """Load data from HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        # Load segments
        segments_data = {}
        if 'segments' in f:
            for key in f['segments'].keys():
                data = f['segments'][key][:]
                if data.dtype.kind == 'S':  # byte string
                    data = data.astype(str)
                # Handle 2D arrays (umap, pca) - store as list of arrays
                if len(data.shape) == 2:
                    segments_data[key] = [row for row in data]
                else:
                    segments_data[key] = data
        segments = pd.DataFrame(segments_data)
        
        # Load files (optional)
        files_data = {}
        if 'files' in f:
            for key in f['files'].keys():
                data = f['files'][key][:]
                if data.dtype.kind == 'S':
                    data = data.astype(str)
                files_data[key] = data
        files = pd.DataFrame(files_data)
        
        # Load parameters (optional)
        parameters = {}
        if 'parameters' in f:
            parameters = dict(f['parameters'].attrs)
        
        # Load UMAP maprange if present
        if 'umap_maprange' in f:
            parameters['umap_maprange'] = f['umap_maprange'][:]
        
        # Load embeddings PCA if present (legacy format in embeddings group)
        if 'embeddings' in f and 'pca' in f['embeddings']:
            pca_data = f['embeddings/pca'][:]
            segments['pca'] = [row for row in pca_data]
        
        # Ensure cluster_id exists
        if 'cluster_id' not in segments.columns:
            segments['cluster_id'] = 0
    
    # Debug output
    print(f"Loaded {len(segments)} segments from HDF5")
    if 'cluster_id' in segments.columns:
        cluster_counts = segments['cluster_id'].value_counts()
        print(f"  Cluster distribution: {dict(cluster_counts.head(10))}")
    
    return FlatData(
        segments=segments,
        files=files,
        parameters=parameters,
        hdf5_path=hdf5_path
    )


def _load_from_parquet_dir(data_dir: Path) -> FlatData:
    """Load data from directory containing parquet and npy files."""
    segments_path = data_dir / 'segments.parquet'
    files_path = data_dir / 'files.parquet'
    params_path = data_dir / 'parameters.json'
    spectrograms_dir = data_dir / 'spectrograms'
    
    if not segments_path.exists():
        raise FileNotFoundError(f"segments.parquet not found in {data_dir}")
    
    segments = pd.read_parquet(segments_path)
    
    if files_path.exists():
        files = pd.read_parquet(files_path)
    else:
        files = pd.DataFrame()
    
    parameters = {}
    if params_path.exists():
        import json
        with open(params_path, 'r') as f:
            parameters = json.load(f)
    
    # Ensure cluster_id exists
    if 'cluster_id' not in segments.columns:
        segments['cluster_id'] = 0
    
    return FlatData(
        segments=segments,
        files=files,
        parameters=parameters,
        spectrograms_dir=spectrograms_dir if spectrograms_dir.exists() else None
    )


def _load_from_parquet_file(parquet_path: Path) -> FlatData:
    """Load from single parquet file (segments only)."""
    segments = pd.read_parquet(parquet_path)
    
    if 'cluster_id' not in segments.columns:
        segments['cluster_id'] = 0
    
    # Try to find spectrograms in parent directory
    spectrograms_dir = parquet_path.parent / 'spectrograms'
    
    return FlatData(
        segments=segments,
        files=pd.DataFrame(),
        parameters={},
        spectrograms_dir=spectrograms_dir if spectrograms_dir.exists() else None
    )


    print(f"Saved to {output_path}")

def save_to_hdf5(data: FlatData, output_path: str, compress: bool = True) -> None:
    """Save FlatData to HDF5 format."""
    output_path = Path(output_path)
    
    compression = 'gzip' if compress else None
    
    with h5py.File(output_path, 'w') as f:
        # Debug output
        print(f"  Saving {len(data.segments)} segments...")
        if 'cluster_id' in data.segments.columns:
            cluster_counts = data.segments['cluster_id'].value_counts()
            print(f"    Cluster distribution: {dict(cluster_counts.head(10))}")
        
        # Save segments
        seg_grp = f.create_group('segments')
        
        # Check for duplicate columns
        saved_cols = set()
        for col in data.segments.columns:
            if col in saved_cols:
                print(f"    Warning: Duplicate segment column skipped: {col}")
                continue
            
            arr = data.segments[col].values
            # Handle columns containing arrays (umap, pca)
            if arr.dtype == object and len(arr) > 0 and isinstance(arr[0], np.ndarray):
                arr = np.stack(arr)
            elif arr.dtype == object:
                arr = arr.astype('S')
            seg_grp.create_dataset(col, data=arr, compression=compression)
            saved_cols.add(col)
        
        # Save files
        files_grp = f.create_group('files')
        saved_files_cols = set()
        for col in data.files.columns:
            if col in saved_files_cols:
                continue
            arr = data.files[col].values
            if arr.dtype == object:
                arr = arr.astype('S')
            files_grp.create_dataset(col, data=arr, compression=compression)
            saved_files_cols.add(col)
        
        # Save parameters
        params_grp = f.create_group('parameters')
        for key, val in data.parameters.items():
            if isinstance(val, (dict, list)):
                continue  # Skip complex types
            if isinstance(val, np.ndarray):
                continue  # Save arrays separately below
            try:
                params_grp.attrs[key] = val
            except TypeError:
                pass  # Skip unsupported types
        
        # Save spectrograms
        spec_grp = f.create_group('spectrograms')
        if data.spectrograms_dir is not None:
            # Parquet mode - load from .npy files
            for spec_file in data.spectrograms_dir.glob('*.npy'):
                file_id = int(spec_file.stem)
                spec = np.load(spec_file)
                spec_grp.create_dataset(str(file_id), data=spec, compression=compression)
        elif data.hdf5_path is not None and data.hdf5_path != output_path:
            # HDF5 mode - copy from source HDF5
            with h5py.File(data.hdf5_path, 'r') as src:
                if 'spectrograms' in src:
                    for key in src['spectrograms'].keys():
                        if key not in spec_grp:
                            spec = src['spectrograms'][key][:]
                            spec_grp.create_dataset(key, data=spec, compression=compression)
        
        # Also save any cached spectrograms that might have been modified
        for file_id, spec in data._spectrogram_cache.items():
            key = str(file_id)
            if key not in spec_grp:
                spec_grp.create_dataset(key, data=spec, compression=compression)
        
        # Copy embeddings from source HDF5 if available
        if data.hdf5_path is not None and data.hdf5_path != output_path:
            with h5py.File(data.hdf5_path, 'r') as src:
                if 'embeddings' in src:
                    emb_grp = f.create_group('embeddings')
                    for key in src['embeddings'].keys():
                        data_arr = src['embeddings'][key][:]
                        emb_grp.create_dataset(key, data=data_arr, compression=compression)
        
        # Save UMAP maprange if available
        if 'umap_maprange' in data.parameters and 'umap_maprange' not in f:
            f.create_dataset('umap_maprange', data=data.parameters['umap_maprange'])

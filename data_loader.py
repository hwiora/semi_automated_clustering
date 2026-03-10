"""
Data loader for UMAP clustering application.
Supports both Parquet/NPY mode and HDF5 mode.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
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
    _spectrogram_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _segment_spec_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _pitch_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    max_spectrogram_cache: int = 2000
    max_segment_spec_cache: int = 20000
    max_pitch_cache: int = 2000

    @staticmethod
    def _cache_get(cache: OrderedDict, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    @staticmethod
    def _cache_put(cache: OrderedDict, key, value, max_size: int) -> None:
        if key in cache:
            cache.pop(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)
    
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
        cached = self._cache_get(self._spectrogram_cache, int(file_id))
        if cached is not None:
            return cached
        
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
                        self._cache_put(self._spectrogram_cache, int(file_id), spec, self.max_spectrogram_cache)
                        return spec
                
                # Also check if key exists directly in spectrograms group
                spec_grp = f['spectrograms']
                for key in [f'file_{file_id:04d}', str(file_id)]:
                    if key in spec_grp:
                        spec = spec_grp[key][:]
                        self._cache_put(self._spectrogram_cache, int(file_id), spec, self.max_spectrogram_cache)
                        return spec
                        
        elif self.spectrograms_dir is not None:
            # Parquet mode
            spec_file = self.spectrograms_dir / f"{file_id}.npy"
            if spec_file.exists():
                spec = np.load(spec_file)
                self._cache_put(self._spectrogram_cache, int(file_id), spec, self.max_spectrogram_cache)
                return spec
        
        return None

    def get_pitch(self, file_id: int):
        """Load pitch trajectory for a file as (time, f0), if available."""
        cached = self._cache_get(self._pitch_cache, int(file_id))
        if cached is not None:
            return cached

        if self.hdf5_path is None:
            return None

        with h5py.File(self.hdf5_path, 'r') as f:
            if 'pitch' not in f:
                return None
            pitch_grp = f['pitch']
            key = str(int(file_id))
            if key not in pitch_grp:
                return None
            file_grp = pitch_grp[key]
            if 'time' not in file_grp or 'f0' not in file_grp:
                return None
            t = file_grp['time'][:]
            f0 = file_grp['f0'][:]

        out = (t, f0)
        self._cache_put(self._pitch_cache, int(file_id), out, self.max_pitch_cache)
        return out
    
    def get_segment_spectrogram(self, segment_id: int, context_sec: float = 0.0) -> Optional[np.ndarray]:
        """Extract spectrogram slice for a specific segment (cached)."""
        segment_id = int(segment_id)
        context_sec = max(0.0, float(context_sec))
        cache_key = (segment_id, context_sec)

        # Check cache first
        cached = self._cache_get(self._segment_spec_cache, cache_key)
        if cached is not None:
            return cached
        
        row = self.segments.iloc[segment_id]
        file_id = int(row['file_id'])
        onset_sec = float(row['onset_sec'])
        duration_sec = float(row['duration_sec'])

        if not (np.isfinite(onset_sec) and np.isfinite(duration_sec)):
            return None
        if duration_sec <= 0:
            return None

        onset_sec = max(0.0, onset_sec - context_sec)
        duration_sec = max(0.0, duration_sec + 2.0 * context_sec)
        
        # Compute samples from seconds using stored sample rate
        sr = self.scanrate
        onset_sample = int(onset_sec * sr)
        duration_samples = int(duration_sec * sr)
        
        full_spec = self.get_spectrogram(file_id)
        if full_spec is None:
            return None
        
        # Convert samples to spectrogram columns
        # hop_length = hop size in samples (e.g., 128)
        hop_length = self.parameters.get(
            'hop_length',
            self.parameters.get('spec_hop_length', self.parameters.get('nonoverlap', 128))
        )
        start_col = onset_sample // hop_length
        end_col = (onset_sample + duration_samples) // hop_length
        
        # Clamp to valid range
        n_cols_total = int(full_spec.shape[1])
        if n_cols_total <= 0:
            return None

        start_col = max(0, start_col)
        end_col = min(n_cols_total, end_col)

        # Ensure at least one column for extremely short segments or minor boundary mismatches
        if start_col >= n_cols_total:
            start_col = n_cols_total - 1
        if end_col <= start_col:
            end_col = min(n_cols_total, start_col + 1)
        
        segment_spec = full_spec[:, start_col:end_col]
        
        # Cache it
        self._cache_put(self._segment_spec_cache, cache_key, segment_spec, self.max_segment_spec_cache)
        return segment_spec

    def get_segment_pitch(self, segment_id: int, context_sec: float = 0.0):
        """Extract pitch trajectory for a segment as (relative_time, f0), if available."""
        row = self.segments.iloc[int(segment_id)]
        file_id = int(row['file_id'])
        onset_sec = float(row['onset_sec'])
        duration_sec = float(row['duration_sec'])
        context_sec = max(0.0, float(context_sec))

        if not (np.isfinite(onset_sec) and np.isfinite(duration_sec)):
            return None
        if duration_sec <= 0:
            return None

        pitch = self.get_pitch(file_id)
        if pitch is None:
            return None

        t, f0 = pitch
        start = max(0.0, onset_sec - context_sec)
        end = onset_sec + duration_sec + context_sec
        mask = np.logical_and(t >= start, t <= end)
        if not np.any(mask):
            return None

        t_seg = t[mask] - start
        f0_seg = f0[mask]
        return t_seg, f0_seg
    
    def precache_all_spectrograms(self, max_segments: Optional[int] = None) -> None:
        """Pre-load all segment spectrograms into cache for faster access."""
        n_segments = self.n_segments if max_segments is None else min(self.n_segments, int(max_segments))
        print(f"Pre-caching {n_segments} spectrograms...")
        for seg_id in range(n_segments):
            self.get_segment_spectrogram(seg_id)
            if (seg_id + 1) % 500 == 0:
                print(f"  Cached {seg_id + 1}/{n_segments}...")
        print(f"Done caching {n_segments} spectrograms.")
    
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


def _sanitize_segments_df(segments: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop segments with invalid timing to avoid downstream indexing/display errors."""
    if segments.empty:
        return segments

    if 'onset_sec' not in segments.columns or 'duration_sec' not in segments.columns:
        return segments

    onset = pd.to_numeric(segments['onset_sec'], errors='coerce').to_numpy(dtype=np.float64)
    duration = pd.to_numeric(segments['duration_sec'], errors='coerce').to_numpy(dtype=np.float64)
    valid = np.logical_and(np.isfinite(onset), np.isfinite(duration))
    valid = np.logical_and(valid, duration > 0)

    dropped = int((~valid).sum())
    if dropped > 0:
        print(f"Warning: Dropped {dropped} invalid segments from {source} (non-finite or non-positive duration).")

    return segments.loc[valid].reset_index(drop=True)


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

        segments = _sanitize_segments_df(segments, str(hdf5_path))
    
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

    segments = _sanitize_segments_df(segments, str(data_dir))
    
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

    segments = _sanitize_segments_df(segments, str(parquet_path))
    
    # Try to find spectrograms in parent directory
    spectrograms_dir = parquet_path.parent / 'spectrograms'
    
    return FlatData(
        segments=segments,
        files=pd.DataFrame(),
        parameters={},
        spectrograms_dir=spectrograms_dir if spectrograms_dir.exists() else None
    )


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

        # Copy pitch tracks from source HDF5 if available
        if data.hdf5_path is not None and data.hdf5_path != output_path:
            with h5py.File(data.hdf5_path, 'r') as src:
                if 'pitch' in src:
                    src_pitch = src['pitch']
                    dst_pitch = f.create_group('pitch')
                    for key in src_pitch.keys():
                        src_obj = src_pitch[key]
                        if isinstance(src_obj, h5py.Dataset):
                            dst_pitch.create_dataset(key, data=src_obj[:], compression=compression)
                        else:
                            dst_sub = dst_pitch.create_group(key)
                            for subkey in src_obj.keys():
                                dst_sub.create_dataset(subkey, data=src_obj[subkey][:], compression=compression)
        
        # Save UMAP maprange if available
        if 'umap_maprange' in data.parameters and 'umap_maprange' not in f:
            f.create_dataset('umap_maprange', data=data.parameters['umap_maprange'])

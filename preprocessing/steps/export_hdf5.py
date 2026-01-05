"""
Step: Export preprocessing results to HDF5 format for clustering_app.
"""
import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_day_string


def export_to_hdf5(subject_name, data_dir, spectrograms_dir, segmentation_dir, 
                   embedding_dir, umap_dir, output_path, 
                   sr=32000, hop_length=128, n_pca_components=100,
                   embedding_style='first_timebin', n_neighbors=100, all_args=None):
    """Export all preprocessing results to HDF5 format for clustering_app.
    
    Args:
        subject_name: Subject identifier.
        data_dir: Directory containing wav files (either directly or in day folders).
        spectrograms_dir: Directory with spectrogram pickle files.
        segmentation_dir: Directory with segmentation pickle files.
        embedding_dir: Directory with embedding pickle files.
        umap_dir: Directory with UMAP results.
        output_path: Path for output HDF5 file.
        sr: Sample rate in Hz.
        hop_length: Hop length used for spectrograms.
        n_pca_components: Number of PCA components to compute from embeddings.
        embedding_style: Embedding style subfolder name.
        n_neighbors: UMAP n_neighbors (for loading the right file).
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    print('Collating data for HDF5 export...')
    
    # Collect all data from preprocessing outputs
    all_days = sorted(get_all_days(data_dir))
    
    # Storage for all segments
    all_file_ids = []
    all_filenames = []
    all_onset_sec = []
    all_duration_sec = []
    all_embeddings = []
    
    # Storage for files
    file_paths = []
    file_id_map = {}  # filename -> file_id
    current_file_id = 0
    
    # Storage for spectrograms (file_id -> spectrogram)
    spectrograms = {}
    
    for day in tqdm(all_days, desc='Loading day data'):
        day_str = get_day_string(day)
        
        # Load segmentation
        seg_path = os.path.join(segmentation_dir, f'{subject_name}_{day_str}_subdir_segmentations.pkl')
        if not os.path.exists(seg_path):
            continue
        seg_dict = pickle.load(open(seg_path, 'rb'))
        
        # Load embeddings
        emb_path = os.path.join(embedding_dir, embedding_style, f'{subject_name}_{day_str}_subdir_embeddings.pkl')
        if not os.path.exists(emb_path):
            continue
        emb_dict = pickle.load(open(emb_path, 'rb'))
        
        # Load spectrograms
        spec_path = os.path.join(spectrograms_dir, f'{subject_name}_{day_str}_subdir_spectrograms.pkl')
        if os.path.exists(spec_path):
            spec_dict = pickle.load(open(spec_path, 'rb'))
            # Index spectrograms by filename
            spec_by_fname = {os.path.basename(f): s for f, s in 
                            zip(spec_dict['filename'], spec_dict['spectrogram'])}
        else:
            spec_by_fname = {}
        
        # Determine wav base dir
        if day is None:
            wav_base_dir = data_dir
        else:
            wav_base_dir = os.path.join(data_dir, day_str)
        
        # Build segment data from embeddings (which has per-segment info)
        for fname, seg_range, embedding in zip(emb_dict['filename'], 
                                                emb_dict['segment_range'], 
                                                emb_dict['embedding']):
            # Get or create file_id
            full_path = os.path.join(wav_base_dir, fname)
            if full_path not in file_id_map:
                file_id_map[full_path] = current_file_id
                file_paths.append(full_path)
                
                # Store spectrogram for this file
                if fname in spec_by_fname:
                    spectrograms[current_file_id] = spec_by_fname[fname]
                
                current_file_id += 1
            
            file_id = file_id_map[full_path]
            onset, offset = seg_range
            duration = offset - onset
            
            all_file_ids.append(file_id)
            all_filenames.append(fname)
            all_onset_sec.append(onset)
            all_duration_sec.append(duration)
            all_embeddings.append(embedding)
    
    if len(all_embeddings) == 0:
        print('No data found to export!')
        return
    
    # Stack embeddings
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f'Total segments: {len(all_embeddings)}')
    print(f'Embedding shape: {embeddings_array.shape}')
    
    # Compute PCA on embeddings
    print(f'Computing PCA ({n_pca_components} components)...')
    pca = PCA(n_components=min(n_pca_components, embeddings_array.shape[1]))
    pca_coords = pca.fit_transform(embeddings_array)
    print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')
    
    # Load UMAP coordinates
    umap_path = os.path.join(umap_dir, f'umap_coordinates_{n_neighbors}neighbors.pickle')
    if os.path.exists(umap_path):
        umap_dict = pickle.load(open(umap_path, 'rb'))
        umap_coords = umap_dict['umap_coordinates']
        print(f'Loaded UMAP coordinates: {umap_coords.shape}')
    else:
        print('Warning: UMAP coordinates not found, using PCA[:2]')
        umap_coords = pca_coords[:, :2]
    
    # Write HDF5
    print(f'Writing HDF5 to {output_path}...')
    with h5py.File(output_path, 'w') as f:
        # Create segments group
        seg_grp = f.create_group('segments')
        seg_grp.create_dataset('segment_id', data=np.arange(len(all_file_ids)))
        seg_grp.create_dataset('file_id', data=np.array(all_file_ids, dtype=np.int32))
        seg_grp.create_dataset('onset_sec', data=np.array(all_onset_sec, dtype=np.float32))
        seg_grp.create_dataset('duration_sec', data=np.array(all_duration_sec, dtype=np.float32))
        seg_grp.create_dataset('umap', data=umap_coords.astype(np.float32))  # Shape: (n_segments, 2)
        seg_grp.create_dataset('cluster_id', data=np.zeros(len(all_file_ids), dtype=np.int32))
        seg_grp.create_dataset('pca', data=pca_coords.astype(np.float32))  # Shape: (n_segments, n_components)
        
        # Create files group
        files_grp = f.create_group('files')
        files_grp.create_dataset('file_id', data=np.arange(len(file_paths), dtype=np.int32))
        # Store only filenames (not full paths)
        filenames = [os.path.basename(p) for p in file_paths]
        filename_bytes = np.array([fn.encode('utf-8') for fn in filenames], dtype='S')
        files_grp.create_dataset('filename', data=filename_bytes)
        
        # Create embeddings group (raw embeddings only, PCA is in segments/pc_*)
        emb_grp = f.create_group('embeddings')
        emb_grp.create_dataset('segment_id', data=np.arange(len(all_file_ids), dtype=np.int32))
        emb_grp.create_dataset('raw', data=embeddings_array, compression='gzip')
        
        # Create spectrograms group
        spec_grp = f.create_group('spectrograms')
        spec_grp.create_dataset('file_id', data=np.array(list(spectrograms.keys()), dtype=np.int32))
        for file_id, spec in tqdm(spectrograms.items(), desc='Writing spectrograms'):
            spec_grp.create_dataset(str(file_id), data=spec, compression='gzip')
        
        # Create parameters group
        params_grp = f.create_group('parameters')
        
        # Store all CLI arguments with step-based prefixes (for organization)
        if all_args is not None:
            args_dict = vars(all_args)
            
            # Define parameter categories and their prefixes
            param_map = {
                # General
                'subject': 'subject',
                'data_dir': 'data_dir',
                'output_dir': 'output_dir',
                'device': 'device',
                
                # Audio/Spectrogram parameters
                'sr': 'audio_sr',
                'n_fft': 'spec_n_fft',
                'hop_length': 'spec_hop_length',
                'spec_min_freq': 'spec_min_freq',
                'spec_max_freq': 'spec_max_freq',
                
                # Pitch parameters
                'pitch_floor': 'pitch_floor',
                
                # Segmentation (WhisperSeg) parameters
                'segmenter_model': 'seg_model',
                'seg_min_freq': 'seg_min_freq',
                'seg_time_step': 'seg_time_step',
                'seg_min_length': 'seg_min_length',
                'seg_eps': 'seg_eps',
                'seg_num_trials': 'seg_num_trials',
                
                # Embedding parameters
                'embedder_model': 'emb_model',
                'emb_min_freq': 'emb_min_freq',
                'emb_time_step': 'emb_time_step',
                'emb_num_trials': 'emb_num_trials',
                'embedding_style': 'emb_style',
                'batch_size': 'emb_batch_size',
                
                # UMAP parameters
                'n_neighbors': 'umap_n_neighbors',
                'train_percentage': 'umap_train_percentage(0-1)',
                
                # PCA/Export parameters
                'n_pca': 'pca_n_components',
            }
            
            for arg_name, param_name in param_map.items():
                if arg_name in args_dict:
                    val = args_dict[arg_name]
                    if val is not None and isinstance(val, (str, int, float, bool)):
                        params_grp.attrs[param_name] = val
        
        # Store key derived parameters (ensure these exist even if all_args not provided)
        params_grp.attrs['sr_processing'] = sr  # Rate used for spectrograms
        # params_grp.attrs['nonoverlap'] = hop_length
        params_grp.attrs['n_pca_components'] = n_pca_components
        params_grp.attrs['n_neighbors'] = n_neighbors
        params_grp.attrs['subject'] = subject_name
        
        # Detect and store original wav sample rate
        if file_paths:
            import wave
            try:
                with wave.open(file_paths[0], 'rb') as wf:
                    params_grp.attrs['sr_original'] = wf.getframerate()
            except Exception:
                params_grp.attrs['sr_original'] = sr  # Fallback to processing rate
        
        # Save UMAP map range for visualization
        umap_min = umap_coords.min(axis=0)
        umap_max = umap_coords.max(axis=0)
        umap_range = np.array([umap_min[0], umap_max[0], umap_min[1], umap_max[1]])
        f.create_dataset('umap_maprange', data=umap_range)
    
    print(f'HDF5 export complete: {output_path}')
    print(f'  - {len(all_file_ids)} segments')
    print(f'  - {len(file_paths)} files')
    print(f'  - {len(spectrograms)} spectrograms')

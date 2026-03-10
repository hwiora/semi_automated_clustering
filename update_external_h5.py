"""
Quick script to update external HDF5 files with:
- embeddings from embeddings.pkl
- default parameters
- strip file paths to names only
- remove onset_sample/duration_samples
- convert pc_0, pc_1... → pca matrix
- convert umap_x, umap_y → umap matrix

Supports bulk mode: process all .h5 files in a bird directory
"""

import h5py
import numpy as np
import pickle
from pathlib import Path
import argparse
import os
import wave

# Default parameters to add
DEFAULT_PARAMS = {
    # General
    'subject': 'R3406',
    'data_dir': 'data',
    'output_dir': 'output',
    'device': 'cuda',
    
    # Audio/Spectrogram
    'audio_sr': 32000,
    'spec_n_fft': 512,
    'spec_hop_length': 128,
    'spec_min_freq': 312,
    'spec_max_freq': 8000,
    
    # Pitch
    'pitch_floor': 312,
    
    # Segmentation
    'seg_model': 'nccratliri/whisperseg-base-animal-vad-ct2',
    'seg_min_freq': 312,
    'seg_time_step': 0.002,
    'seg_min_length': 0.01,
    'seg_eps': 0.02,
    'seg_num_trials': 3,
    
    # Embeddings
    'emb_model': 'Systran/faster-whisper-large-v3',
    'emb_min_freq': 0,
    'emb_time_step': 0.0025,
    'emb_num_trials': 3,
    'emb_style': 'first_timebin',
    'emb_batch_size': 32,
    
    # UMAP
    'umap_n_neighbors': 100,
    'umap_train_percentage(0-1)': 1.0,
    
    # PCA
    'pca_n_components': 100,
    
    # Derived parameters (added during export)
    'sr_processing': 32000,
    'sr_original': 44100,
    'n_neighbors': 100,
    'n_pca_components': 100,
}


def update_single_h5(h5_file, filenames, embeddings, subject_name=None, wav_sr=None):
    """Update a single HDF5 file."""
    print(f"\nUpdating {h5_file.name}...")
    
    temp_out = h5_file.with_suffix('.h5.tmp')
    
    # Make a copy of params and update subject
    params = DEFAULT_PARAMS.copy()
    if subject_name:
        params['subject'] = subject_name
    
    # Update sample rate if provided
    if wav_sr:
        params['audio_sr'] = wav_sr
        params['sr_processing'] = wav_sr
        params['sr_original'] = wav_sr
        print(f"  Using detected sample rate: {wav_sr}")
    
    with h5py.File(h5_file, 'r') as f_in, h5py.File(temp_out, 'w') as f_out:
        # Copy segments (excluding onset_sample, duration_samples, converting pc/umap)
        if 'segments' in f_in:
            seg_in = f_in['segments']
            seg_out = f_out.create_group('segments')
            
            skip_keys = {'onset_sample', 'duration_samples'}
            
            # Convert umap_x, umap_y → umap matrix
            if 'umap_x' in seg_in and 'umap_y' in seg_in:
                umap = np.column_stack([seg_in['umap_x'][:], seg_in['umap_y'][:]])
                seg_out.create_dataset('umap', data=umap.astype(np.float32))
                skip_keys.add('umap_x')
                skip_keys.add('umap_y')
                print(f"  ✓ Converted umap_x, umap_y → umap matrix")
            
            # Convert pc_0, pc_1, ... → pca matrix
            pc_cols = sorted([k for k in seg_in.keys() if k.startswith('pc_')],
                           key=lambda x: int(x.split('_')[1]))
            if pc_cols:
                pca = np.column_stack([seg_in[col][:] for col in pc_cols])
                seg_out.create_dataset('pca', data=pca.astype(np.float32))
                skip_keys.update(pc_cols)
                print(f"  ✓ Converted {len(pc_cols)} PC columns → pca matrix")
            
            # Check if this is an unclustered file (reset cluster_id to 0)
            is_unclustered = h5_file.name.endswith('_unclustered.h5')
            
            # Copy remaining datasets
            for key in seg_in.keys():
                if key not in skip_keys:
                    if key == 'cluster_id' and is_unclustered:
                        # Reset cluster_id to 0 for unclustered files
                        n_segments = len(seg_in[key][:])
                        seg_out.create_dataset(key, data=np.zeros(n_segments, dtype=np.int32))
                        print(f"  ✓ Reset cluster_id to 0 (unclustered file)")
                    else:
                        seg_out.create_dataset(key, data=seg_in[key][:])
            print(f"  ✓ Copied segments")
        
        # Copy files group (preserve original order, just strip to basenames)
        # OR rebuild if empty/missing
        files_copied = False
        if 'files' in f_in:
            files_in = f_in['files']
            # Check if it has any datasets
            if len(files_in.keys()) > 0:
                files_out = f_out.create_group('files')
                
                for key in files_in.keys():
                    data = files_in[key][:]
                    if key == 'path':
                        # Strip to basename and rename to 'filename'
                        if data.dtype.kind == 'S':
                            data = np.array([Path(p.decode('utf-8')).name.encode('utf-8') for p in data], dtype='S')
                        else:
                            data = np.array([Path(str(p)).name for p in data])
                        files_out.create_dataset('filename', data=data)
                    else:
                        files_out.create_dataset(key, data=data)
                
                # Add file_id if missing
                if 'file_id' not in files_in:
                    n_files = 0
                    if 'path' in files_in:
                        n_files = len(files_in['path'])
                    elif len(files_in.keys()) > 0:
                        first_key = list(files_in.keys())[0]
                        n_files = len(files_in[first_key])
                    
                    if n_files > 0:
                        files_out.create_dataset('file_id', data=np.arange(n_files, dtype=np.int32))
                
                print(f"  ✓ Copied files group (stripped paths to basenames)")
                files_copied = True
        
        # Fallback: Rebuild files group if copy failed or was empty
        if not files_copied and filenames:
            files_out = f_out.create_group('files') if 'files' not in f_out else f_out['files']
            files_out.create_dataset('file_id', data=np.arange(len(filenames), dtype=np.int32))
            filename_bytes = np.array([fn.encode('utf-8') for fn in filenames], dtype='S')
            files_out.create_dataset('filename', data=filename_bytes)
            print(f"  ✓ Rebuilt files group from wavs ({len(filenames)} files) - original empty/missing")
        
        # Add embeddings
        emb_out = f_out.create_group('embeddings')
        if embeddings is not None:
            emb_out.create_dataset('segment_id', data=np.arange(len(embeddings), dtype=np.int32))
            emb_out.create_dataset('raw', data=embeddings, compression='gzip')
            print(f"  ✓ Added embeddings ({embeddings.shape})")
        elif 'embeddings' in f_in:
            for key in f_in['embeddings'].keys():
                emb_out.create_dataset(key, data=f_in['embeddings'][key][:], compression='gzip')
            print(f"  ✓ Copied existing embeddings")
        
        # Copy spectrograms
        if 'spectrograms' in f_in:
            spec_out = f_out.create_group('spectrograms')
            for key in f_in['spectrograms'].keys():
                spec_out.create_dataset(key, data=f_in['spectrograms'][key][:], compression='gzip')
            print(f"  ✓ Copied spectrograms")
        
        # Add parameters
        params_out = f_out.create_group('parameters')
        for key, val in params.items():
            params_out.attrs[key] = val
        print(f"  ✓ Added {len(params)} parameters")
        
        # Copy other datasets
        for key in f_in.keys():
            if key not in ['segments', 'files', 'embeddings', 'spectrograms', 'parameters']:
                if isinstance(f_in[key], h5py.Dataset):
                    f_out.create_dataset(key, data=f_in[key][:])
    
    # Replace original
    os.remove(h5_file)
    temp_out.rename(h5_file)
    print(f"  ✅ Done!")


def main():
    parser = argparse.ArgumentParser(description='Update external HDF5 files to new format')
    parser.add_argument('h5_path', help='Path to HDF5 file or directory (for bulk mode)')
    parser.add_argument('wavs_dir', help='Path to directory containing wav files (used for subject name)')
    parser.add_argument('embeddings_pkl', nargs='?', default=None, 
                        help='Path to embeddings.pkl file (optional)')
    parser.add_argument('--bulk', '-b', action='store_true',
                        help='Bulk mode: process all .h5 files in h5_path directory')
    
    args = parser.parse_args()
    
    h5_path = Path(args.h5_path).resolve()
    wavs_dir = Path(args.wavs_dir).resolve()
    embeddings_pkl = Path(args.embeddings_pkl).resolve() if args.embeddings_pkl else None
    
    # Check wavs_dir exists (just for validation and subject name)
    if not wavs_dir.exists():
        print(f"ERROR: Directory not found: {wavs_dir}")
        return
        
    # Extract subject name from path
    subject_name = wavs_dir.parent.name  # e.g., R3406
    print(f"Subject: {subject_name}")
    
    # Get wav files for SR detection
    wav_files = sorted(wavs_dir.glob('*.wav'))
    if not wav_files:
        print(f"WARNING: No .wav files found in {wavs_dir}. Cannot detect sample rate.")
    
    # Detect sample rate from first file
    wav_sr = None
    filenames = []
    if wav_files:
        filenames = [f.name for f in wav_files]
        try:
            with wave.open(str(wav_files[0]), 'rb') as w:
                wav_sr = w.getframerate()
            print(f"Detected sample rate from {wav_files[0].name}: {wav_sr} Hz")
        except Exception as e:
            print(f"WARNING: Could not detect sample rate: {e}")
    
    # Load embeddings
    embeddings = None
    if embeddings_pkl and embeddings_pkl.exists():
        with open(embeddings_pkl, 'rb') as f:
            emb_data = pickle.load(f)
        if isinstance(emb_data, dict) and 'embedding' in emb_data:
            embeddings = np.array(emb_data['embedding'], dtype=np.float32)
        else:
            embeddings = np.array(emb_data, dtype=np.float32)
        print(f"Loaded embeddings: {embeddings.shape}")
    
    if args.bulk:
        # Bulk mode: process all .h5 files in directory
        if not h5_path.is_dir():
            print(f"ERROR: For bulk mode, h5_path must be a directory: {h5_path}")
            return
        
        h5_files = sorted(h5_path.glob('*.h5'))
        if not h5_files:
            print(f"ERROR: No .h5 files found in {h5_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"BULK MODE: Processing {len(h5_files)} HDF5 files for {subject_name}")
        print(f"Sample Rate: {wav_sr} Hz")
        print(f"{'='*60}")
        
        for h5_file in h5_files:
            update_single_h5(h5_file, filenames, embeddings, subject_name, wav_sr)
        
        print(f"\n{'='*60}")
        print(f"✅ Completed {len(h5_files)} files!")
        print(f"{'='*60}\n")
    else:
        # Single file mode
        if not h5_path.is_file():
            print(f"ERROR: File not found: {h5_path}")
            return
        
        update_single_h5(h5_path, filenames, embeddings, subject_name, wav_sr)


if __name__ == '__main__':
    main()

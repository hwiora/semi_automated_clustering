"""
Update HDF5 files to new structure without recomputing data.

This script takes an old HDF5 file and updates it to the new format by:
1. Converting umap_x, umap_y → umap matrix
2. Converting pc_0, pc_1, ... → pca matrix
3. Adding file_id to files and spectrograms groups
4. Adding segment_id to embeddings group
5. Updating parameters with organized names
6. Loading embeddings from .pkl file
7. Stripping file paths to filenames only
8. Removing onset_sample and duration_samples
"""

import h5py
import numpy as np
import pickle
from pathlib import Path
import argparse


def update_hdf5_file(input_path, output_path=None, dry_run=False, embeddings_pkl=None, default_params=None):
    """Update an HDF5 file to the new structure.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file (default: overwrites input)
        dry_run: If True, only print what would be changed
        embeddings_pkl: Path to embeddings.pkl file (optional)
        default_params: Dict of default parameters to add (optional)
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    print(f"\n{'='*60}")
    print(f"Updating: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")
    
    # Read old file
    with h5py.File(input_path, 'r') as f_in:
        # Prepare new structure
        changes = []
        
        # Check segments group
        if 'segments' in f_in:
            seg = f_in['segments']
            
            # Check for umap_x, umap_y → umap conversion
            if 'umap_x' in seg and 'umap_y' in seg and 'umap' not in seg:
                changes.append("✓ Convert umap_x, umap_y → umap matrix")
            
            # Check for pc_0, pc_1, ... → pca conversion
            pc_cols = [k for k in seg.keys() if k.startswith('pc_')]
            if pc_cols and 'pca' not in seg:
                changes.append(f"✓ Convert {len(pc_cols)} PC columns → pca matrix")
        
        # Check files group
        if 'files' in f_in:
            files = f_in['files']
            if 'file_id' not in files:
                changes.append("✓ Add file_id to files group")
        
        # Check spectrograms group
        if 'spectrograms' in f_in:
            spec = f_in['spectrograms']
            if 'file_id' not in spec:
                n_specs = len([k for k in spec.keys() if k != 'file_id'])
                changes.append(f"✓ Add file_id list to spectrograms ({n_specs} files)")
        
        # Check embeddings group
        if 'embeddings' in f_in:
            emb = f_in['embeddings']
            if 'segment_id' not in emb:
                changes.append("✓ Add segment_id to embeddings group")
        
        # Check parameters
        if 'parameters' in f_in:
            params = dict(f_in['parameters'].attrs)
            
            # Check for old parameter names
            old_names = ['sr', 'n_fft', 'hop_length', 'n_neighbors']
            new_names = ['audio_sr', 'spec_n_fft', 'spec_hop_length', 'umap_n_neighbors']
            needs_rename = any(old in params and new not in params for old, new in zip(old_names, new_names))
            
            if needs_rename:
                changes.append("✓ Reorganize parameters with step-based prefixes")
            
            if 'nonoverlap' in params and 'hop_length' not in params:
                changes.append("✓ Rename nonoverlap → hop_length")
        
        if not changes:
            print("✅ File is already in the new format!")
            return
        
        print("Changes to be made:")
        for change in changes:
            print(f"  {change}")
        print()
        
        if dry_run:
            print("Dry run complete. Use --apply to make changes.")
            return
        
        # Perform the update
        print("Applying changes...")
        
        # Create new file (or temp file if overwriting)
        import tempfile
        if output_path == input_path:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.h5', dir=output_path.parent)
            import os
            os.close(temp_fd)
            temp_path = Path(temp_path)
        else:
            temp_path = output_path
        
        with h5py.File(temp_path, 'w') as f_out:
            # Copy and transform segments
            if 'segments' in f_in:
                seg_in = f_in['segments']
                seg_out = f_out.create_group('segments')
                
                # Track which columns to skip
                skip_cols = set()
                
                # Convert umap_x, umap_y → umap
                if 'umap_x' in seg_in and 'umap_y' in seg_in:
                    umap_x = seg_in['umap_x'][:]
                    umap_y = seg_in['umap_y'][:]
                    umap = np.column_stack([umap_x, umap_y])
                    seg_out.create_dataset('umap', data=umap.astype(np.float32))
                    skip_cols.add('umap_x')
                    skip_cols.add('umap_y')
                    print("  ✓ Converted umap_x, umap_y → umap")
                
                # Convert pc_0, pc_1, ... → pca
                pc_cols = sorted([k for k in seg_in.keys() if k.startswith('pc_')],
                                key=lambda x: int(x.split('_')[1]))
                if pc_cols:
                    pc_data = np.column_stack([seg_in[col][:] for col in pc_cols])
                    seg_out.create_dataset('pca', data=pc_data.astype(np.float32))
                    skip_cols.update(pc_cols)
                    print(f"  ✓ Converted {len(pc_cols)} PC columns → pca")
                
                # Copy remaining datasets (skip unwanted fields)
                for key in seg_in.keys():
                    if key not in skip_cols and key not in ['onset_sample', 'duration_samples']:
                        seg_out.create_dataset(key, data=seg_in[key][:])
            
            # Copy and update files
            if 'files' in f_in:
                files_in = f_in['files']
                files_out = f_out.create_group('files')
                
                # Add file_id if missing
                if 'path' in files_in:
                    n_files = len(files_in['path'])
                    files_out.create_dataset('file_id', data=np.arange(n_files, dtype=np.int32))
                    print(f"  ✓ Added file_id to files ({n_files} files)")
                
                # Copy other datasets (strip paths to basenames)
                for key in files_in.keys():
                    data = files_in[key][:]
                    if key == 'path':
                        # Strip to basename only
                        if data.dtype.kind == 'S':  # byte strings
                            data = np.array([Path(p.decode('utf-8')).name.encode('utf-8') for p in data], dtype='S')
                        else:  # regular strings
                            data = np.array([Path(p).name for p in data])
                        print(f"  ✓ Stripped paths to filenames only ({len(data)} files)")
                    files_out.create_dataset(key, data=data)
            
            # Copy and update embeddings
            if 'embeddings' in f_in:
                emb_in = f_in['embeddings']
                emb_out = f_out.create_group('embeddings')
                
                # Add segment_id if missing
                if 'raw' in emb_in:
                    n_segments = emb_in['raw'].shape[0]
                    emb_out.create_dataset('segment_id', data=np.arange(n_segments, dtype=np.int32))
                    print(f"  ✓ Added segment_id to embeddings ({n_segments} segments)")
                
                # Copy datasets
                for key in emb_in.keys():
                    emb_out.create_dataset(key, data=emb_in[key][:], compression='gzip')
            
            # Copy and update spectrograms
            if 'spectrograms' in f_in:
                spec_in = f_in['spectrograms']
                spec_out = f_out.create_group('spectrograms')
                
                # Collect file IDs
                file_ids = sorted([int(k) for k in spec_in.keys() if k.isdigit()])
                if file_ids:
                    spec_out.create_dataset('file_id', data=np.array(file_ids, dtype=np.int32))
                    print(f"  ✓ Added file_id to spectrograms ({len(file_ids)} spectrograms)")
                
                # Copy spectrograms
                for key in spec_in.keys():
                    spec_out.create_dataset(key, data=spec_in[key][:], compression='gzip')
            
            # Update parameters
            if 'parameters' in f_in:
                params_in = dict(f_in['parameters'].attrs)
                params_out = f_out.create_group('parameters')
                
                # Parameter renaming map
                rename_map = {
                    'sr': 'audio_sr',
                    'n_fft': 'spec_n_fft',
                    'hop_length': 'spec_hop_length',
                    'nonoverlap': 'spec_hop_length',  # Also handles old nonoverlap
                    'spec_min_freq': 'spec_min_freq',
                    'spec_max_freq': 'spec_max_freq',
                    'pitch_floor': 'pitch_floor',
                    'n_neighbors': 'umap_n_neighbors',
                    'n_pca_components': 'pca_n_components',
                    'scanrate': 'sr_processing',  # Handle old name
                }
                
                renamed = set()
                for old_name, new_name in rename_map.items():
                    if old_name in params_in and new_name not in params_in:
                        params_out.attrs[new_name] = params_in[old_name]
                        renamed.add(old_name)
                
                # Copy remaining parameters
                for key, val in params_in.items():
                    if key not in renamed and key not in rename_map.values():
                        params_out.attrs[key] = val
                
                if renamed:
                    print(f"  ✓ Renamed {len(renamed)} parameters")
            
            # Copy other top-level datasets (like umap_maprange)
            for key in f_in.keys():
                if key not in ['segments', 'files', 'embeddings', 'spectrograms', 'parameters']:
                    if isinstance(f_in[key], h5py.Dataset):
                        f_out.create_dataset(key, data=f_in[key][:])
    
    # File is now closed - safe to replace
    if output_path == input_path:
        import os
        # Windows-safe replacement: delete original first
        if output_path.exists():
            os.remove(str(output_path))
        temp_path.rename(output_path)
    
    print(f"\n✅ Successfully updated: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Update HDF5 files to new structure')
    parser.add_argument('input', help='Input HDF5 file or directory')
    parser.add_argument('--output', help='Output path (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without making changes')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Process all .h5 files in directory recursively')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        update_hdf5_file(input_path, args.output, args.dry_run)
    elif input_path.is_dir():
        # Directory
        pattern = '**/*.h5' if args.recursive else '*.h5'
        h5_files = list(input_path.glob(pattern))
        
        if not h5_files:
            print(f"No .h5 files found in {input_path}")
            return
        
        print(f"Found {len(h5_files)} HDF5 file(s)\n")
        
        for h5_file in h5_files:
            output = None
            if args.output:
                output = Path(args.output) / h5_file.relative_to(input_path)
                output.parent.mkdir(parents=True, exist_ok=True)
            
            update_hdf5_file(h5_file, output, args.dry_run)
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()

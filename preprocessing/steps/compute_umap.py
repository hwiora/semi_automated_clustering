"""
Step 6: Compute UMAP dimensionality reduction on embeddings.
"""
import os
import pickle
import random
from glob import glob

import numpy as np
import umap
from tqdm import tqdm


def collate_embedding_files(embedding_dir):
    """Collate all daily embedding files into single arrays.
    
    Args:
        embedding_dir: Directory containing embedding pickle files.
        
    Returns:
        Tuple of (filenames, segment_ranges, embeddings) lists.
    """
    embedding_files = sorted(glob(os.path.join(embedding_dir, '*.pkl')))
    
    all_fnames = []
    all_ranges = []
    all_embs = []
    
    for fpath in embedding_files:
        data = pickle.load(open(fpath, 'rb'))
        all_fnames.extend(data['filename'])
        all_ranges.extend(data['segment_range'])
        all_embs.extend(data['embedding'])
    
    return all_fnames, all_ranges, all_embs


def compute_umap(embedding_dir, output_dir, n_neighbors=100, 
                 train_percentage=1.0, sample_seed=22):
    """Compute UMAP projection from embeddings.
    
    Args:
        embedding_dir: Directory containing embedding pickle files.
        output_dir: Directory to save UMAP results.
        n_neighbors: Number of neighbors for UMAP.
        train_percentage: Fraction of data to use for fitting (1.0 = all).
        sample_seed: Random seed for sampling if train_percentage < 1.0.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print('Collating embeddings...')
    all_fnames, all_ranges, all_embs = collate_embedding_files(embedding_dir)
    
    if len(all_embs) == 0:
        print('No embeddings found!')
        return
    
    print(f'Total embeddings: {len(all_embs)}')
    
    all_embs = np.asarray(all_embs, dtype=np.float32)

    if train_percentage <= 0 or train_percentage > 1.0:
        raise ValueError(f'train_percentage must be in (0, 1], got {train_percentage}')

    if train_percentage == 1.0:
        print('Computing UMAP (fitting on all data)...')
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_jobs=-1).fit(all_embs)
        umap_coords = reducer.embedding_
    else:
        random.seed(sample_seed)
        n_samples = max(1, int(train_percentage * len(all_embs)))
        sampled_idx = sorted(random.sample(range(len(all_embs)), k=n_samples))
        sampled_embs = all_embs[sampled_idx]
        
        print(f'Computing UMAP (fitting on {n_samples} samples)...')
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_jobs=-1).fit(sampled_embs)
        
        print('Transforming all embeddings...')
        umap_coords = reducer.transform(all_embs)
    
    # Save UMAP model
    model_path = os.path.join(output_dir, f'umap_model_{n_neighbors}neighbors.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump(reducer, f)
    print(f'Saved UMAP model: {model_path}')
    
    # Save coordinates
    umap_dict = {
        'filename': all_fnames,
        'segment_range': all_ranges,
        'umap_coordinates': umap_coords
    }
    coords_path = os.path.join(output_dir, f'umap_coordinates_{n_neighbors}neighbors.pickle')
    with open(coords_path, 'wb') as f:
        pickle.dump(umap_dict, f)
    print(f'Saved UMAP coordinates: {coords_path}')
    
    print('UMAP computation complete!')


def compute_umap_for_embeddings(embeddings, n_neighbors=100,
                                train_percentage=1.0, sample_seed=22):
    """Compute UMAP projection directly from an embedding matrix.

    Standalone version — accepts a numpy array, returns coordinates.

    Args:
        embeddings: np.ndarray of shape (n_segments, embedding_dim).
        n_neighbors: Number of neighbors for UMAP.
        train_percentage: Fraction of data used to fit UMAP (1.0 = all data).
        sample_seed: Random seed used when train_percentage < 1.0.

    Returns:
        np.ndarray of shape (n_segments, 2) with UMAP coordinates.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f'embeddings must be 2-D, got shape {embeddings.shape}')
    if train_percentage <= 0 or train_percentage > 1.0:
        raise ValueError(f'train_percentage must be in (0, 1], got {train_percentage}')

    n_neighbors = min(n_neighbors, len(embeddings) - 1)

    if train_percentage == 1.0:
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_jobs=-1).fit(embeddings)
        return reducer.embedding_
    else:
        random.seed(sample_seed)
        n_samples = max(1, int(train_percentage * len(embeddings)))
        sampled_idx = sorted(random.sample(range(len(embeddings)), k=n_samples))
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_jobs=-1).fit(embeddings[sampled_idx])
        return reducer.transform(embeddings)

def save_umap_results(save_path, filenames, onsets_per_file, offsets_per_file,
                      umap_coords):
    """Save UMAP results to a pickle file readable by MATLAB.

    One UMAP run covers many audio files. Each file contributes a variable
    number of segments. The pickle stores one entry per segment:
        filename  – repeated for each segment belonging to that file
        segment_range – [onset, offset] in seconds per segment
        umap_coordinates – (N_total, 2) numpy array

    Can be loaded in MATLAB via:
        umap_data = py.pickle.load(py.open(umap_file, 'rb'));

    Args:
        save_path: Path to the output .pickle file.
        filenames: List of audio filenames (one per file).
        onsets_per_file: List of arrays/lists of onset times (seconds),
                         one array per file.
        offsets_per_file: List of arrays/lists of offset times (seconds),
                          one array per file.
        umap_coords: np.ndarray of shape (N_total_segments, 2).
    """
    all_filenames = []
    all_segment_ranges = []
    for fname, onsets, offsets in zip(filenames, onsets_per_file,
                                     offsets_per_file):
        for on, off in zip(onsets, offsets):
            all_filenames.append(fname)
            all_segment_ranges.append([float(on), float(off)])

    data = {
        'filename': all_filenames,
        'segment_range': all_segment_ranges,
        'umap_coordinates': np.asarray(umap_coords, dtype=np.float32),
    }
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

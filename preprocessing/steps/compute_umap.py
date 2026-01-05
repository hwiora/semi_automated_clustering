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
    embedding_files = glob(os.path.join(embedding_dir, '*.pkl'))
    
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
    
    if train_percentage == 1.0:
        print('Computing UMAP (fitting on all data)...')
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_jobs=-1).fit(all_embs)
        umap_coords = reducer.embedding_
    else:
        random.seed(sample_seed)
        n_samples = int(train_percentage * len(all_embs))
        sampled_idx = sorted(random.sample(range(len(all_embs)), k=n_samples))
        sampled_embs = [all_embs[i] for i in sampled_idx]
        
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

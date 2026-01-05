"""
Preprocessing steps for audio data.

Steps are imported lazily to avoid dependency errors when running --help.
"""

__all__ = [
    'compute_spectrograms', 
    'compute_pitch',
    'run_whisperseg',
    'compute_embeddings',
    'compute_umap',
    'export_to_hdf5',
]


def __getattr__(name):
    """Lazy import of step functions."""
    if name == 'compute_spectrograms':
        from .compute_spectrogram import compute_spectrograms
        return compute_spectrograms
    elif name == 'compute_pitch':
        from .compute_pitch import compute_pitch
        return compute_pitch
    elif name == 'run_whisperseg':
        from .run_whisperseg import run_whisperseg
        return run_whisperseg
    elif name == 'compute_embeddings':
        from .compute_embeddings import compute_embeddings
        return compute_embeddings
    elif name == 'compute_umap':
        from .compute_umap import compute_umap
        return compute_umap
    elif name == 'export_to_hdf5':
        from .export_hdf5 import export_to_hdf5
        return export_to_hdf5
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

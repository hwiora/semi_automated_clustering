# Semi-Automated Clustering

Interactive Python application for clustering and annotating high-dimensional data using UMAP embeddings and visual inspection. Originally designed for vocalization analysis but applicable to any time-series segmentation data.

## Overview

This repository contains two main components:

1. **Preprocessing Pipeline** (`preprocessing/`) - Converts raw audio recordings into HDF5 format ready for clustering
2. **Clustering App** (`clustering_app.py`) - Interactive tool for semi-automated cluster annotation

## Features

### Preprocessing Pipeline
- **Spectrogram computation** - Convert audio to spectrograms using librosa
- **Pitch extraction** - Extract pitch contours from audio files using Parselmouth (Praat)
- **Segmentation** - Detect vocalizations using [WhisperSeg](https://github.com/nianlonggu/WhisperSeg)
- **Embedding extraction** - Generate neural embeddings for each vocalization using Whisper
- **UMAP projection** - Reduce embedding dimensionality for visualization
- **HDF5 export** - Package everything for the clustering app

### Clustering App
- **Interactive UMAP Visualization**: Click to select clusters of points in embedding space
- **Blob Detection**: Automatic density-based grouping with adjustable threshold and radius
- **Spectrogram Viewer**: View and navigate spectrograms sorted by various criteria
- **Human-in-the-Loop Workflow**:
  - **CLUSTERING**: View and edit final cluster assignments
  - **BLOBBING**: Select and assign groups using density-based detection
  - **PROOFREADING**: Review and refine assignments before finalizing
- **Multiple Sort Modes**: Timestamp, duration, random, nearest-neighbor chain, outlier-first
- **HDF5 Export/Import**: Save and reload your work

---   

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hwiora/semi_automated_clustering.git
cd semi_automated_clustering
```

### 2. Create conda environment

```bash
conda create -n clustering python=3.10
conda activate clustering
```

### 3. Install dependencies

```bash
# Install clustering app dependencies
pip install -r requirements.txt

# Install preprocessing dependencies
pip install -r preprocessing/requirements.txt

# Install WhisperSeg (modified version)
pip install git+https://github.com/hwiora/WhisperSeg.git
```

---

## Quick Start

### Option 1: Full Pipeline (Raw Audio → Clustering)

```bash
# Step 1: Preprocess your audio data
cd preprocessing
python preprocess.py --subject MY_SUBJECT \
    --data-dir /path/to/audio/data \
    --output-dir /path/to/output \
    --all

# Step 2: Run the clustering app
cd ..
python clustering_app.py /path/to/output/MY_SUBJECT.h5
```

### Option 2: Use Existing HDF5 Data

```bash
python clustering_app.py path/to/your/data.h5
```

---

## Preprocessing Pipeline

The preprocessing pipeline converts raw audio recordings into HDF5 format.

### Input Data Format

Your audio data should be organized in subdirectories (e.g., typically day folders, but can be any name):

```
data/
├── 001/           # Day 1 (or any subfolder name)
│   ├── file1.wav
│   └── file2.wav
├── subfolder_B/   # Another subfolder
│   └── ...
└── ...
```

### Usage

```bash
cd preprocessing

# Run all steps (recommended for first-time use):
python preprocess.py --subject MY_SUBJECT \
    --data-dir /path/to/audio \
    --output-dir /path/to/output \
    --all \
    --device cuda

# Or run individual steps:
python preprocess.py --subject MY_SUBJECT \
    --data-dir /path/to/audio \
    --output-dir /path/to/output \
    --compute-spectrograms \
    --compute-segmentations \
    --compute-embeddings \
    --compute-umap \
    --export-hdf5
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sr` | 32000 | Sample rate (Hz) |
| `--segmenter-model` | nccratliri/whisperseg-base-animal-vad-ct2 | WhisperSeg model |
| `--n-neighbors` | 100 | UMAP neighbors |
| `--n-pca` | 10 | PCA components to include |
| `--device` | cuda | Device for ML models (cuda or cpu) |
| `--hdf5-output` | {output_dir}/{subject}.h5 | Custom HDF5 output path |

Run `python preprocess.py --help` for all options.

---

## Clustering App

### Keyboard Controls

| Key | Action |
|-----|--------|
| `=` / `-` | Adjust blob detection threshold |
| `↑` / `↓` | Adjust blob detection radius |
| `←` / `→` | Navigate spectrograms |
| `PageUp/Down` | Scroll grid rows |
| `Home` / `End` | Jump to start/end of current group |
| `w` | Assign selected blob |
| `m` | Move segment to different cluster |
| `u` | Merge clusters |
| `o` | Change sort mode |
| `c` | Cycle modes (CLUSTERING → BLOBBING → PROOFREADING) |
| `Shift+C` | Reset all assignments |
| `x` | Export to HDF5 |
| `q` | Quit |

### Workflow

1. **Start in CLUSTERING mode**: View existing cluster assignments
2. **Press `c` to enter BLOBBING mode**: 
   - Click on UMAP points to select blobs
   - Adjust threshold/radius as needed
   - Press `w` to assign selected blob
3. **Press `c` to enter PROOFREADING mode**:
   - Review precluster assignments
   - Use `m` to move segments, `u` to merge preclusters
4. **Press `c` and confirm to finalize**:
   - Returns to CLUSTERING mode with new assignments
5. **Press `x` to export**: Save your work to HDF5

---

## HDF5 Data Format

The HDF5 file structure used by both the preprocessing pipeline and clustering app:

```
data.h5
├── segments/                    # Segment metadata
│   ├── segment_id               # Segment index
│   ├── file_id                  # Which source file
│   ├── onset_sec                # Segment start time (seconds)
│   ├── duration_sec             # Segment duration (seconds)
│   ├── umap                     # UMAP coordinates (n_segments, 2)
│   ├── pca                      # PCA coordinates (n_segments, n_components)
│   └── cluster_id               # Cluster assignment
├── files/                       # Source file metadata
│   ├── file_id                  # File index (matches segments/file_id)
│   └── filename                 # File names
├── embeddings/                  # Embedding vectors
│   ├── segment_id               # Segment index (matches segments/segment_id)
│   └── raw                      # Full embedding vectors (n_segments, 1280)
├── spectrograms/                # Spectrogram data
│   ├── file_id                  # List of file IDs with spectrograms
│   ├── 0                        # Spectrogram for file_id=0
│   ├── 1                        # Spectrogram for file_id=1
│   └── ...
├── parameters/                  # Processing parameters (organized by step)
│   └── attrs:
│       ├── subject, data_dir, output_dir, device
│       ├── audio_sr, spec_n_fft, spec_hop_length, spec_min_freq, spec_max_freq
│       ├── pitch_floor
│       ├── seg_model, seg_min_freq, seg_time_step, seg_min_length, seg_eps, seg_num_trials
│       ├── emb_model, emb_min_freq, emb_time_step, emb_num_trials, emb_style, emb_batch_size
│       ├── umap_n_neighbors, umap_train_percentage
│       ├── pca_n_components
│       ├── sr_processing, sr_original  # Derived parameters
│       └── ...
└── umap_maprange               # UMAP axis limits
```

---

## Example Data

Sample data is available on Zenodo: https://doi.org/10.5281/zenodo.18156254

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{semi_automated_clustering,
  author = {Lee, Kanghwi},
  title = {Semi-Automated Clustering},
  url = {https://github.com/hwiora/semi_automated_clustering}
}
```

For the WhisperSeg segmentation model:

```bibtex
@inproceedings{gu2024whisperseg,
  title={Positive Transfer of the Whisper Speech Transformer to Human and Animal Voice Activity Detection},
  author={Gu, Nianlong and others},
  booktitle={ICASSP 2024},
  year={2024}
}
```

---

## License

MIT License

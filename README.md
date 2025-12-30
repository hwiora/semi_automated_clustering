# Semi-Automated Clustering

Interactive Python application for clustering and annotating high-dimensional data using UMAP embeddings and visual inspection. Originally designed for vocalization analysis but applicable to any time-series segmentation data.

## Features

- **Interactive UMAP Visualization**: Click to select clusters of points in embedding space
- **Blob Detection**: Automatic density-based grouping with adjustable threshold and radius
- **Spectrogram Viewer**: View and navigate spectrograms sorted by various criteria
- **Human-in-the-Loop Workflow**:
  - **CLUSTERING**: View and edit final cluster assignments
  - **BLOBBING**: Select and assign groups using density-based detection
  - **PROOFREADING**: Review and refine assignments before finalizing
- **Multiple Sort Modes**: Timestamp, duration, random, nearest-neighbor chain, outlier-first
- **HDF5 Export/Import**: Save and reload your work

## Installation

```bash
# Clone the repository
git clone https://github.com/mesahwi/semi_automated_clustering.git
cd semi_automated_clustering

# Create conda environment (recommended)
conda create -n clustering python=3.10
conda activate clustering

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python -m semi_automated_clustering path/to/your/data.h5
```

## Keyboard Controls

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

## Data Format

The application uses HDF5 files with the following structure:

```
data.h5
├── segments/                    # Segment metadata
│   ├── file_id                  # Which source file
│   ├── onset_sec                # Segment start time (seconds)
│   ├── duration_sec             # Segment duration (seconds)
│   ├── onset_sample             # Start sample index
│   ├── duration_samples         # Duration in samples
│   ├── umap_x, umap_y           # UMAP coordinates
│   ├── cluster_id               # Cluster assignment
│   └── pc_0, pc_1, ...          # PCA embeddings (optional)
├── files/                       # Source file metadata
│   └── path                     # File paths
├── spectrograms/                # Spectrogram data
│   ├── 0                        # Full spectrogram for file 0
│   ├── 1                        # Full spectrogram for file 1
│   └── ...
├── parameters/                  # Processing parameters
│   └── attrs: scanrate, nonoverlap, ...
└── umap_maprange               # UMAP axis limits (optional)
```

## Workflow

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

## License

MIT License

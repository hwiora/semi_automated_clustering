"""
Interactive UMAP Clustering Application.

Keyboard Controls:
    =/-     : Adjust threshold (×1.2 / ÷1.2)
    ↑/↓     : Adjust radius (±0.5)
    ←/→     : Navigate spectrograms in Fig 20/33
    w       : Assign selected blob (turns gray, excluded from future blobs)
    m       : Move selected spectrogram to cluster (dialog)
    u       : Merge another cluster into current cluster (dialog)
    o       : Select sorting mode (dialog)
    c       : Cycle modes (CLUSTERING → BLOBBING → PROOFREADING → CLUSTERING)
    Shift+C : Reset all clusters
    x       : Export to HDF5 file
    q       : Quit
"""

import argparse
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid Qt event loop issues
import matplotlib.pyplot as plt
# Disable default key bindings that conflict with our app
plt.rcParams['keymap.save'] = []        # Was 's', 'ctrl+s'
plt.rcParams['keymap.fullscreen'] = []  # Was 'f', 'ctrl+f'
plt.rcParams['keymap.home'] = []        # Was 'h', 'r', 'home'
plt.rcParams['keymap.back'] = []        # Was 'left', 'c', 'backspace'
plt.rcParams['keymap.forward'] = []     # Was 'right', 'v'
plt.rcParams['keymap.pan'] = []         # Was 'p'
plt.rcParams['keymap.zoom'] = []        # Was 'o'
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List, Dict, Set

try:
    from .data_loader import load_data, save_to_hdf5, FlatData
    from .blob_detector import BlobDetector, Blob
    from .spectrogram_viewer import SpectrogramViewer
except ImportError:
    from data_loader import load_data, save_to_hdf5, FlatData
    from blob_detector import BlobDetector, Blob
    from spectrogram_viewer import SpectrogramViewer


class ClusteringApp:
    """Interactive UMAP clustering application."""
    
    def __init__(self, data: FlatData):
        self.data = data
        
        umap_params = data.parameters.get('umap', {})
        self.n_pixels = umap_params.get('num_pixels', 1200)
        
        self.detector = BlobDetector(
            n_pixels=self.n_pixels,
            radius=umap_params.get('radius_kernel', 20.0),
            theta=umap_params.get('theta', 0.1),
            max_clust=umap_params.get('max_clust', 30),
        )
        
        # Always calculate maprange from actual data to ensure all points are covered
        # (stored maprange may be stale if coordinates were updated via CSV)
        coords = data.umap_coords
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        margin_x = (max_x - min_x) * 0.05
        margin_y = (max_y - min_y) * 0.05
        self.maprange = np.array([
            [min_x - margin_x, min_y - margin_y],
            [max_x + margin_x, max_y + margin_y]
        ])
        
        # App mode: 'CLUSTERING', 'BLOBBING', 'PROOFREADING'
        self.app_mode = 'CLUSTERING'  # Start in cluster viewing mode
        
        # State
        self.current_blob_id = 1  # Next blob ID to assign (B1, B2,...)
        self.selected_blob: Optional[Blob] = None
        self.selected_segment: Optional[int] = None
        self.shift_pressed = False
        
        # Blob assignments (BLOBBING mode): B1, B2,... → segment_ids
        self.blob_assignments: Dict[int, np.ndarray] = {}
        self.assigned_segment_indices: Set[int] = set()
        
        # Precluster assignments (PROOFREADING mode): P1, P2,... → segment_ids
        self.precluster_assignments: Dict[int, np.ndarray] = {}
        
        # Final clusters are stored in data.segments['cluster_id'] (C1, C2,...)
        
        # Input mode for number entry (None, 'move', 'merge', 'sort', 'nearest_space', 'finalize')
        self.input_mode: Optional[str] = None
        self.input_buffer: str = ""
        
        # Spectrogram Viewer state
        self.fig20_segments: List[int] = []
        self.fig20_offset: int = 0
        self.fig20_selected: Optional[int] = None
        self.fig20_sort_mode = 'timestamp'  # 'timestamp', 'duration', 'random', 'nn_chain_umap/pc', 'nn_outlier_umap/pc'
        self.fig20_display_mode = 'sorted'  # 'sorted' or 'umap_neighbors'
        self.fig20_max_duration: Optional[float] = None  # For consistent time scale
        
        # Blob/Precluster Grid state
        self.fig33_selected: Optional[int] = None
        self.fig33_current_group: Optional[int] = None  # Which blob/precluster is displayed in Fig20
        self.fig33_row_offset: int = 0  # Scroll offset for rows
        self.fig33_max_rows: int = 10    # Max rows to display at once
        self._fig33_sorted_cache: Dict[int, np.ndarray] = {}  # Cached sorted segments per group
        
        # Cluster Grid state
        self.fig34_selected: Optional[int] = None
        self.fig34_current_cluster: Optional[int] = None
        self.fig34_row_offset: int = 0  # Scroll offset for rows
        self.fig34_max_rows: int = 10    # Max rows to display at once
        self._fig34_sorted_cache: Dict[int, np.ndarray] = {}  # Cached sorted segments per cluster
        
        # Track active figure for scrolling (33 or 34)
        self.active_figure: int = 33  # Default to Fig33
        
        # Pre-cache all spectrograms for faster updates
        self.data.precache_all_spectrograms()
        
        self._setup_figures()
        self._update_blobs()
    
    def _setup_figures(self) -> None:
        """Set up matplotlib figures."""
        # Disable matplotlib's default keyboard shortcuts to avoid conflicts
        # plt.rcParams['keymap.save'] = []  # 's' 
        plt.rcParams['keymap.quit'] = []  # 'q'
        plt.rcParams['keymap.fullscreen'] = []  # 'f'
        plt.rcParams['keymap.home'] = []  # 'h', 'r'
        plt.rcParams['keymap.back'] = []  # 'left', 'c', 'backspace'
        plt.rcParams['keymap.forward'] = []  # 'right', 'v'
        plt.rcParams['keymap.pan'] = []  # 'p'
        plt.rcParams['keymap.zoom'] = []  # 'o'
        plt.rcParams['keymap.grid'] = []  # 'g'
        plt.rcParams['keymap.yscale'] = []  # 'l'
        plt.rcParams['keymap.xscale'] = []  # 'k'
        
        # UMAP Scatter
        self.fig_umap, self.ax_umap = plt.subplots(figsize=(8, 8), num="UMAP Scatter")
        self.fig_umap.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig_umap.canvas.mpl_connect('key_release_event', self._on_key_release)
        self.fig_umap.canvas.mpl_connect('button_press_event', self._on_umap_click)
        
        try:
            self.fig_umap.canvas.manager.window.wm_geometry("+0+300")
        except:
            pass
        
        # Spectrogram Viewer: Selected blob spectrograms (dynamic axes based on duration)
        self.fig_spec = plt.figure(figsize=(20, 2.5), num="Spectrogram Viewer")
        self.spec_viewer = SpectrogramViewer(self.data, fig=self.fig_spec, n_rows=1, n_cols=12)
        self.fig_spec.canvas.mpl_connect('button_press_event', self._on_fig20_click)
        self.fig_spec.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        try:
            self.fig_spec.canvas.manager.window.wm_geometry("+0+0")
        except:
            pass
        
        # Blob/Precluster Grid: Visualizer (B1,B2... or P1,P2...)
        self.fig_blobs = plt.figure(figsize=(16, 8), num="Blob/Precluster Grid")
        self.fig_blobs.canvas.mpl_connect('button_press_event', self._on_fig33_click)
        self.fig_blobs.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        try:
            self.fig_blobs.canvas.manager.window.wm_geometry("+850+0")
        except:
            pass
        
        # Cluster Grid: Final cluster visualizer (C1,C2...)
        self.fig_clusters = plt.figure(figsize=(16, 8), num="Cluster Grid")
        self.fig_clusters.canvas.mpl_connect('button_press_event', self._on_fig34_click)
        self.fig_clusters.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        try:
            self.fig_clusters.canvas.manager.window.wm_geometry("+850+500")
        except:
            pass
        
        # Colormap
        n_colors = 31
        colors = np.zeros((n_colors, 4))
        colors[0] = [0.0, 0.0, 0.3, 1.0]
        colors[1:] = plt.cm.jet(np.linspace(0, 1, n_colors - 1))
        self.blob_cmap = ListedColormap(colors)
        
        # Initialize Cluster Grid with existing clusters
        self._update_cluster_visualizer()
    
    def _get_active_umap_coords(self) -> tuple:
        """Get UMAP coords excluding assigned segments."""
        all_coords = self.data.umap_coords
        if not self.assigned_segment_indices:
            return all_coords, np.arange(len(all_coords))
        
        mask = np.ones(len(all_coords), dtype=bool)
        mask[list(self.assigned_segment_indices)] = False
        active_indices = np.where(mask)[0]
        return all_coords[active_indices], active_indices
    
    def _update_blobs(self) -> None:
        """Recompute blob detection."""
        active_coords, self._active_indices = self._get_active_umap_coords()
        if len(active_coords) > 0:
            self.detector.detect_blobs(active_coords, self.maprange)
        else:
            # Clear blob image when no active points remain
            self.detector._blob_image = None
            self.detector._blobs = []
        self._update_display()
    
    def _update_blobs_threshold_only(self) -> None:
        self.detector.update_threshold_only()
        self._update_display()
    
    def _update_blobs_radius_only(self) -> None:
        self.detector.update_radius_only()
        self._update_display()
    
    def _update_display(self) -> None:
        """Update UMAP Scatter."""
        self.ax_umap.clear()
        umap_coords = self.data.umap_coords
        
        # Only show blob overlay in BLOBBING mode
        if self.app_mode == 'BLOBBING' and self.detector.blob_image is not None and self.detector.maprange is not None:
            # Use detector's maprange to ensure coordinate mapping matches
            mr = self.detector.maprange
            extent = [mr[0, 0], mr[1, 0], mr[0, 1], mr[1, 1]]
            self.ax_umap.imshow(
                self.detector.blob_image, origin='lower', extent=extent,
                cmap=self.blob_cmap, interpolation='nearest', vmin=0, vmax=30
            )
        
        if self.app_mode == 'BLOBBING':
            # BLOBBING: Assigned = gray, Active = white
            if self.assigned_segment_indices:
                assigned_coords = umap_coords[list(self.assigned_segment_indices)]
                self.ax_umap.scatter(assigned_coords[:, 0], assigned_coords[:, 1],
                                    c='gray', s=5, alpha=0.6)
            
            active_coords, _ = self._get_active_umap_coords()
            if len(active_coords) > 0:
                self.ax_umap.scatter(active_coords[:, 0], active_coords[:, 1],
                                    c='white', s=3, alpha=0.8)
        else:
            # PROOFREADING/CLUSTERING: All points white
            self.ax_umap.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                c='white', s=3, alpha=0.8)
        
        # Selected
        if self.selected_segment is not None:
            seg_coords = umap_coords[self.selected_segment]
            self.ax_umap.scatter([seg_coords[0]], [seg_coords[1]],
                                s=100, facecolors='none', edgecolors='red', linewidths=3)
        
        mode_str = f"[{self.app_mode}]"
        self.ax_umap.set_title(f"{mode_str} r={self.detector.radius:.1f}, θ={self.detector.theta:.2e}")
        self.ax_umap.set_xlabel("UMAP 1")
        self.ax_umap.set_ylabel("UMAP 2")
        
        # Gray out UMAP Scatter in non-BLOBBING modes
        if self.app_mode == 'BLOBBING':
            self.ax_umap.set_facecolor((0.0, 0.0, 0.3))
        else:
            self.ax_umap.set_facecolor((0.2, 0.2, 0.2))  # Grayed out
            self.ax_umap.set_title(f"{mode_str} (View Only)")
        
        # Set axis limits to match maprange to prevent clipping
        if self.maprange is not None:
            self.ax_umap.set_xlim(self.maprange[0, 0], self.maprange[1, 0])
            self.ax_umap.set_ylim(self.maprange[0, 1], self.maprange[1, 1])
        
        self.fig_umap.canvas.draw_idle()
    
    def _sort_segments(self, segment_indices: np.ndarray) -> np.ndarray:
        """Sort segments based on current sort mode."""
        if len(segment_indices) == 0:
            return segment_indices
        
        if self.fig20_sort_mode == 'timestamp':
            onsets = self.data.segments.iloc[segment_indices]['onset_sec'].values
            return segment_indices[np.argsort(onsets)]
        elif self.fig20_sort_mode == 'duration':
            durations = self.data.segments.iloc[segment_indices]['duration_sec'].values
            return segment_indices[np.argsort(durations)]
        elif self.fig20_sort_mode == 'random':
            return np.random.permutation(segment_indices)
        elif self.fig20_sort_mode == 'nn_chain_umap':
            return self._nn_chain_sort(segment_indices, 'umap')
        elif self.fig20_sort_mode == 'nn_chain_pc':
            return self._nn_chain_sort(segment_indices, 'pc')
        elif self.fig20_sort_mode == 'nn_outlier_umap':
            return self._nn_outlier_sort(segment_indices, 'umap')
        elif self.fig20_sort_mode == 'nn_outlier_pc':
            return self._nn_outlier_sort(segment_indices, 'pc')
        return segment_indices
    
    def _get_coords(self, segment_indices: np.ndarray, space: str) -> np.ndarray:
        """Get coordinates for segments in specified space."""
        if space == 'umap':
            return self.data.umap_coords[segment_indices]
        else:  # pc
            if self.data.pc_coords is None:
                print("Warning: No PC coordinates available, using UMAP instead")
                return self.data.umap_coords[segment_indices]
            n_components = min(100, self.data.pc_coords.shape[1])
            return self.data.pc_coords[segment_indices, :n_components]
    
    def _nn_chain_sort(self, segment_indices: np.ndarray, space: str) -> np.ndarray:
        """Sort by greedy nearest neighbor chain (adjacent = similar)."""
        if len(segment_indices) <= 1:
            return segment_indices
        
        coords = self._get_coords(segment_indices, space)
        n = len(segment_indices)
        visited = np.zeros(n, dtype=bool)
        order = []
        
        # Start with first element
        current = 0
        for _ in range(n):
            order.append(current)
            visited[current] = True
            
            if len(order) < n:
                # Find nearest unvisited
                distances = np.sum((coords - coords[current])**2, axis=1)
                distances[visited] = np.inf
                current = np.argmin(distances)
        
        return segment_indices[order]
    
    def _nn_outlier_sort(self, segment_indices: np.ndarray, space: str, k: int = 8) -> np.ndarray:
        """Sort by sum of k-NN distances (typical first, outliers last). FlatClust-style."""
        if len(segment_indices) <= 1:
            return segment_indices
        
        coords = self._get_coords(segment_indices, space)
        n = len(segment_indices)
        
        # Compute pairwise distances
        # For each point, sum distances to k nearest neighbors
        k = min(k, n - 1)  # Can't have more neighbors than points
        
        # Compute all pairwise distances
        dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=2))
        
        # Sort each row and sum k smallest (excluding self at 0)
        sorted_dists = np.sort(dist_matrix, axis=1)
        nn_sums = np.sum(sorted_dists[:, 1:k+1], axis=1)  # Skip first (self distance = 0)
        
        # Sort by sum: smallest sum = most typical (first), largest sum = outlier (last)
        order = np.argsort(nn_sums)
        return segment_indices[order]
    
    def _update_fig20(self) -> None:
        """Update Spectrogram Viewer."""
        if not self.fig20_segments:
            return
        
        start = self.fig20_offset
        remaining_segs = self.fig20_segments[start:]
        if not remaining_segs:
            return
        
        # Calculate how many segments fit using SpectrogramViewer.FIXED_WIDTH_PER_COL
        # Available width is ~0.96, so max ~480 columns total
        AVAILABLE_WIDTH = 0.94  # Leave some margin
        GAP = 0.003
        MAX_COLS = int(AVAILABLE_WIDTH / SpectrogramViewer.FIXED_WIDTH_PER_COL)
        
        # Get column counts for remaining segments
        col_counts = []
        for seg_id in remaining_segs:
            spec = self.data.get_segment_spectrogram(seg_id)
            col_counts.append(spec.shape[1] if spec is not None and spec.size > 0 else 0)
        
        # Find how many segments fit in available width
        cumsum = np.cumsum(col_counts)
        n_display = np.searchsorted(cumsum, MAX_COLS) + 1
        n_display = max(4, min(n_display, 50))  # Clamp to 4-50 range
        
        end = min(start + n_display, len(self.fig20_segments))
        display_segs = self.fig20_segments[start:end]
        
        # Display spectrograms - viewer uses same FIXED_WIDTH_PER_COL
        self.spec_viewer.display_with_highlight(display_segs, self.fig20_selected)
        
        # Set title AFTER display (since display clears the figure)
        if self.fig34_current_cluster:
            group_str = f"C{self.fig34_current_cluster}"
        elif self.fig33_current_group:
            prefix = "B" if self.app_mode == 'BLOBBING' else "P"
            group_str = f"{prefix}{self.fig33_current_group}"
        else:
            group_str = "Selected"
        
        # Show appropriate sort description based on display mode
        if self.fig20_display_mode == 'umap_neighbors':
            sort_desc = "nearest neighbors (UMAP)"
        else:
            sort_desc = self.fig20_sort_mode
        
        self.fig_spec.suptitle(f"{group_str} | Sort: {sort_desc} | "
                               f"[{start+1}-{end}/{len(self.fig20_segments)}]", fontsize=10)
        
        self.fig_spec.canvas.draw_idle()
        self.fig_spec.canvas.flush_events()
    
    def _update_blob_visualizer(self) -> None:
        """Update Blob/Precluster Grid with blobs (BLOBBING) or preclusters (PROOFREADING)."""
        self.fig_blobs.clear()
        self._blob_fig_axes = {}
        
        # Choose data source based on mode
        if self.app_mode == 'BLOBBING':
            data_to_show = self.blob_assignments
            prefix = "B"
            title = "BLOBBING"
            empty_msg = "No blobs assigned\nClick blob in Fig32 + press 'w'"
        else:  # PROOFREADING or CLUSTERING
            data_to_show = self.precluster_assignments
            prefix = "P"
            title = "PROOFREADING"
            empty_msg = "No preclusters"
        
        if not data_to_show:
            self.fig_blobs.text(0.5, 0.5, empty_msg, ha='center', va='center', fontsize=14)
            self.fig_blobs.suptitle(f"{title}", fontsize=12)
            self.fig_blobs.canvas.draw_idle()
            return
        
        total_rows = len(data_to_show)
        max_specs_per_row = 15
        
        # Apply row paging
        sorted_group_ids = sorted(data_to_show.keys())
        start_row = self.fig33_row_offset
        end_row = min(start_row + self.fig33_max_rows, total_rows)
        visible_group_ids = sorted_group_ids[start_row:end_row]
        n_visible_rows = len(visible_group_ids)
        
        if n_visible_rows == 0:
            return
        
        # First pass: find max total duration across ALL rows (for consistent scale across pages)
        max_total_duration = 0
        all_row_data = {}
        for group_id in sorted_group_ids:  # ALL groups, not just visible
            segment_indices = data_to_show[group_id]
            # Use cached sorted segments, or create cache if missing
            if group_id not in self._fig33_sorted_cache:
                self._fig33_sorted_cache[group_id] = self._sort_segments(segment_indices)
            sorted_indices = self._fig33_sorted_cache[group_id]
            n_segs = min(len(sorted_indices), max_specs_per_row)
            if n_segs > 0:
                durations = self.data.segments.iloc[sorted_indices[:n_segs]]['duration_sec'].values
                total_dur = durations.sum()
                max_total_duration = max(max_total_duration, total_dur)
                all_row_data[group_id] = (sorted_indices[:n_segs], durations, sorted_indices)
        
        # Filter to visible rows only
        row_data = {gid: all_row_data[gid] for gid in visible_group_ids if gid in all_row_data}
        
        if max_total_duration == 0:
            return
        
        # Second pass: render with global scale
        available_width = 0.88
        for row_idx, group_id in enumerate(visible_group_ids):
            if group_id not in row_data:
                continue
            display_indices, durations, full_sorted_indices = row_data[group_id]
            n_segs = len(display_indices)
            total_dur = durations.sum()
            row_width = (total_dur / max_total_duration) * available_width
            
            left = 0.07
            # Layout: top=0.92, bottom=0.03, distribute max_rows evenly
            top_margin = 0.92
            bottom_margin = 0.03
            total_height = top_margin - bottom_margin  # 0.89
            row_spacing = total_height / self.fig33_max_rows
            height = row_spacing * 0.85  # 85% of slot for content
            bottom = top_margin - (row_idx + 1) * row_spacing + (row_spacing - height) / 2
            
            x_pos = left
            gap = 0.003  # Gap between spectrograms
            for col_idx in range(n_segs):
                seg_id = display_indices[col_idx]
                width = (durations[col_idx] / total_dur) * row_width
                box_width = max(width - gap, 0.006)  # Minimum box size
                
                ax = self.fig_blobs.add_axes([x_pos, bottom, box_width, height])
                
                spec = self.data.get_segment_spectrogram(seg_id)
                if spec is not None and spec.size > 0:
                    ax.imshow(spec[::-1, :], aspect='auto', cmap='inferno', interpolation='nearest')
                
                ax.set_xticks([])
                ax.set_yticks([])
                # Store FULL sorted list for click handling
                self._blob_fig_axes[id(ax)] = (seg_id, group_id, full_sorted_indices)
                
                # Removed red box highlighting from Fig33 - only show in Fig20
                
                if col_idx == 0:
                    label = f"{prefix}{group_id} ({len(full_sorted_indices)})"
                    ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center')
                
                x_pos += box_width + gap  # Advance by actual box width + gap
        
        # Title with scroll info
        scroll_info = f" [{start_row+1}-{end_row}/{total_rows}]" if total_rows > self.fig33_max_rows else ""
        self.fig_blobs.suptitle(f"{title}{scroll_info}", fontsize=12)
        self.fig_blobs.canvas.draw_idle()
    
    def _update_cluster_visualizer(self) -> None:
        """Update Cluster Grid with final clusters (C1, C2,...)."""
        self.fig_clusters.clear()
        self._cluster_fig_axes = {}
        
        # Build cluster assignments from data.segments['cluster_id']
        cluster_data = {}
        for idx, row in self.data.segments.iterrows():
            cid = int(row['cluster_id'])
            if cid not in cluster_data:
                cluster_data[cid] = []
            cluster_data[cid].append(idx)
        
        # Convert to numpy arrays
        for cid in cluster_data:
            cluster_data[cid] = np.array(cluster_data[cid])
        
        if not cluster_data:
            self.fig_clusters.text(0.5, 0.5, "No clusters", ha='center', va='center', fontsize=14)
            self.fig_clusters.suptitle("CLUSTERS", fontsize=12)
            self.fig_clusters.canvas.draw_idle()
            return
        
        total_rows = len(cluster_data)
        max_specs_per_row = 15
        
        # Apply row paging
        sorted_cluster_ids = sorted(cluster_data.keys())
        start_row = self.fig34_row_offset
        end_row = min(start_row + self.fig34_max_rows, total_rows)
        visible_cluster_ids = sorted_cluster_ids[start_row:end_row]
        n_visible_rows = len(visible_cluster_ids)
        
        if n_visible_rows == 0:
            return
        
        # First pass: find max total duration across ALL rows (for consistent scale across pages)
        max_total_duration = 0
        all_row_data = {}
        for clust_id in sorted_cluster_ids:  # ALL clusters, not just visible
            segment_indices = cluster_data[clust_id]
            # Use cached sorted segments, or create cache if missing
            if clust_id not in self._fig34_sorted_cache:
                self._fig34_sorted_cache[clust_id] = self._sort_segments(segment_indices)
            sorted_indices = self._fig34_sorted_cache[clust_id]
            n_segs = min(len(sorted_indices), max_specs_per_row)
            if n_segs > 0:
                durations = self.data.segments.iloc[sorted_indices[:n_segs]]['duration_sec'].values
                total_dur = durations.sum()
                max_total_duration = max(max_total_duration, total_dur)
                all_row_data[clust_id] = (sorted_indices[:n_segs], durations, sorted_indices)
        
        # Filter to visible rows only
        row_data = {cid: all_row_data[cid] for cid in visible_cluster_ids if cid in all_row_data}
        
        if max_total_duration == 0:
            return
        
        # Second pass: render with global scale
        available_width = 0.88
        for row_idx, clust_id in enumerate(visible_cluster_ids):
            if clust_id not in row_data:
                continue
            display_indices, durations, full_sorted_indices = row_data[clust_id]
            n_segs = len(display_indices)
            total_dur = durations.sum()
            row_width = (total_dur / max_total_duration) * available_width
            
            left = 0.07
            # Layout: top=0.92, bottom=0.03, distribute max_rows evenly
            top_margin = 0.92
            bottom_margin = 0.03
            total_height = top_margin - bottom_margin  # 0.89
            row_spacing = total_height / self.fig34_max_rows
            height = row_spacing * 0.85  # 85% of slot for content
            bottom = top_margin - (row_idx + 1) * row_spacing + (row_spacing - height) / 2
            
            x_pos = left
            gap = 0.003  # Gap between spectrograms
            for col_idx in range(n_segs):
                seg_id = display_indices[col_idx]
                width = (durations[col_idx] / total_dur) * row_width
                box_width = max(width - gap, 0.006)  # Minimum box size
                
                ax = self.fig_clusters.add_axes([x_pos, bottom, box_width, height])
                
                spec = self.data.get_segment_spectrogram(seg_id)
                if spec is not None and spec.size > 0:
                    ax.imshow(spec[::-1, :], aspect='auto', cmap='inferno', interpolation='nearest')
                
                ax.set_xticks([])
                ax.set_yticks([])
                # Store FULL sorted list for click handling
                self._cluster_fig_axes[id(ax)] = (seg_id, clust_id, full_sorted_indices)
                
                # Removed cyan box highlighting from Fig34 - only show in Fig20
                
                if col_idx == 0:
                    ax.set_ylabel(f"C{clust_id} ({len(full_sorted_indices)})", fontsize=8, rotation=0, ha='right', va='center')
                
                x_pos += box_width + gap  # Advance by actual box width + gap
        
        # Title with scroll info
        scroll_info = f" [{start_row+1}-{end_row}/{total_rows}]" if total_rows > self.fig34_max_rows else ""
        self.fig_clusters.suptitle(f"CLUSTERS{scroll_info}", fontsize=12)
        self.fig_clusters.canvas.draw_idle()
    
    def _on_fig20_click(self, event) -> None:
        if event.inaxes is None:
            return
        seg_id = self.spec_viewer.get_clicked_segment(event)
        if seg_id is not None:
            self.fig20_selected = seg_id
            self.fig33_selected = seg_id
            self.selected_segment = seg_id  # For UMAP red circle
            print(f">>> Selected segment {seg_id} (press 'm' to move)")
            self._update_fig20()
            self._update_display()  # Update UMAP with red circle
    
    def _on_fig33_click(self, event) -> None:
        if event.inaxes is None:
            return
        
        ax_id = id(event.inaxes)
        if hasattr(self, '_blob_fig_axes') and ax_id in self._blob_fig_axes:
            seg_id, group_id, group_segments = self._blob_fig_axes[ax_id]
            
            # Track previous state
            prev_group = self.fig33_current_group
            prev_selected = self.fig33_selected
            
            self.fig33_selected = seg_id
            self.fig20_selected = seg_id
            self.selected_segment = seg_id  # For UMAP red circle
            self.active_figure = 33  # Set active for scrolling
            
            # Prefix based on mode
            prefix = "B" if self.app_mode == 'BLOBBING' else "P"
            
            # Only update Fig20 if switching to a different group
            if prev_group != group_id:
                self.fig33_current_group = group_id
                self.fig34_current_cluster = None  # Clear cluster selection
                # Use cached sorted segments (or create cache if missing)
                if group_id not in self._fig33_sorted_cache:
                    self._fig33_sorted_cache[group_id] = self._sort_segments(group_segments)
                self.fig20_segments = self._fig33_sorted_cache[group_id].tolist()
                self.fig20_offset = 0
                self.fig20_display_mode = 'sorted'  # Using sort mode
                print(f">>> Switched to {prefix}{group_id}, showing segment {seg_id}")
                self._update_fig20()
                self._update_blob_visualizer()
            elif prev_selected != seg_id:
                # Same group, different segment - just update Fig20
                if seg_id in self.fig20_segments:
                    seg_idx = self.fig20_segments.index(seg_id)
                    n_display = 12
                    if seg_idx < self.fig20_offset or seg_idx >= self.fig20_offset + n_display:
                        self.fig20_offset = max(0, seg_idx - n_display // 2)
                print(f">>> Selected segment {seg_id} from {prefix}{group_id} (press 'm' to move)")
                self._update_fig20()
            
            # Always update UMAP to show red circle
            self._update_display()
    
    def _on_fig34_click(self, event) -> None:
        """Handle click on Cluster Grid."""
        if event.inaxes is None:
            return
        
        ax_id = id(event.inaxes)
        if hasattr(self, '_cluster_fig_axes') and ax_id in self._cluster_fig_axes:
            seg_id, clust_id, cluster_segments = self._cluster_fig_axes[ax_id]
            
            prev_cluster = self.fig34_current_cluster
            prev_selected = self.fig34_selected
            
            self.fig34_selected = seg_id
            self.fig20_selected = seg_id
            self.selected_segment = seg_id  # For UMAP red circle
            self.active_figure = 34  # Set active for scrolling
            
            if prev_cluster != clust_id:
                self.fig34_current_cluster = clust_id
                self.fig33_current_group = None  # Clear precluster selection
                # Use cached sorted segments (or create cache if missing)
                if clust_id not in self._fig34_sorted_cache:
                    self._fig34_sorted_cache[clust_id] = self._sort_segments(cluster_segments)
                self.fig20_segments = self._fig34_sorted_cache[clust_id].tolist()
                self.fig20_offset = 0
                self.fig20_display_mode = 'sorted'  # Using sort mode
                print(f">>> Switched to C{clust_id}, showing segment {seg_id}")
                self._update_fig20()
                self._update_cluster_visualizer()
            elif prev_selected != seg_id:
                if seg_id in self.fig20_segments:
                    seg_idx = self.fig20_segments.index(seg_id)
                    n_display = 12
                    if seg_idx < self.fig20_offset or seg_idx >= self.fig20_offset + n_display:
                        self.fig20_offset = max(0, seg_idx - n_display // 2)
                print(f">>> Selected segment {seg_id} from C{clust_id}")
                self._update_fig20()
            
            # Always update UMAP to show red circle
            self._update_display()
    
    def _on_umap_click(self, event) -> None:
        if event.inaxes != self.ax_umap:
            return
        
        segment_id = self._find_nearest_segment(event.xdata, event.ydata)
        
        if segment_id is None:
            self.selected_segment = None
            self.selected_blob = None
            self._update_display()
            return
        
        self.selected_segment = segment_id
        row = self.data.segments.iloc[segment_id]
        print(f"\n>>> Segment {segment_id}: File {int(row['file_id'])}, "
              f"Onset {row['onset_sec']:.3f}s, C{int(row['cluster_id'])}")
        
        if self.app_mode == 'BLOBBING':
            # Show blob segments (sorted by distance to clicked point)
            blob = self._find_active_blob(event.xdata, event.ydata)
            if blob is not None:
                self.selected_blob = blob
                
                # Guard: check if any active indices remain
                if len(self._active_indices) == 0:
                    print(f"    All segments already assigned!")
                    self.selected_blob = None
                elif len(blob.segment_indices) == 0:
                    print(f"    Blob {blob.blob_id}: Empty blob (stale data)")
                    self.selected_blob = None
                else:
                    all_blob_segs = self._active_indices[blob.segment_indices]
                    blob_segs_original = np.array([s for s in all_blob_segs if s not in self.assigned_segment_indices])
                    n_remaining = len(blob_segs_original)
                    n_total = len(all_blob_segs)
                    
                    if n_remaining == 0:
                        print(f"    Blob {blob.blob_id}: All {n_total} segments already assigned!")
                        self.selected_blob = None
                    else:
                        print(f"    Blob {blob.blob_id}: {n_remaining}/{n_total} segments remaining")
                        # Sort blob segments by distance to clicked point
                        umap_coords = self.data.umap_coords
                        distances = np.sqrt(np.sum((umap_coords[blob_segs_original] - 
                                                   umap_coords[segment_id])**2, axis=1))
                        sorted_blob_segs = blob_segs_original[np.argsort(distances)]
                        
                        self.fig20_segments = sorted_blob_segs.tolist()
                        self.fig20_offset = 0
                        self.fig20_selected = segment_id
                        self.fig20_display_mode = 'umap_neighbors'  # Sorted by UMAP distance
                        self.fig33_current_group = None
                        self.fig34_current_cluster = None
                        self._update_fig20()
            else:
                self.selected_blob = None
                # No blob clicked - show 20 nearest neighbors
                umap_coords = self.data.umap_coords
                distances = np.sqrt(np.sum((umap_coords - umap_coords[segment_id])**2, axis=1))
                nearest_20 = np.argsort(distances)[:20]
                print(f"    Showing 20 nearest UMAP neighbors (no blob)")
                
                self.fig20_segments = nearest_20.tolist()
                self.fig20_offset = 0
                self.fig20_selected = segment_id
                self.fig20_display_mode = 'umap_neighbors'
                self.fig33_current_group = None
                self.fig34_current_cluster = None
                self._update_fig20()
        
        elif self.app_mode == 'PROOFREADING':
            # Show 20 nearest UMAP neighbors (sorted by distance, not by sort mode)
            umap_coords = self.data.umap_coords
            distances = np.sqrt(np.sum((umap_coords - umap_coords[segment_id])**2, axis=1))
            nearest_20 = np.argsort(distances)[:20]  # 20 nearest including self
            print(f"    Showing 20 nearest UMAP neighbors")
            
            self.fig20_segments = nearest_20.tolist()  # Already sorted by distance
            self.fig20_offset = 0
            self.fig20_selected = segment_id
            self.fig20_display_mode = 'umap_neighbors'  # Showing UMAP neighbors
            self.fig33_current_group = None
            self.fig34_current_cluster = None
            self._update_fig20()
        
        else:  # CLUSTERING mode
            # Show 20 nearest UMAP neighbors (sorted by distance, not by sort mode)
            umap_coords = self.data.umap_coords
            distances = np.sqrt(np.sum((umap_coords - umap_coords[segment_id])**2, axis=1))
            nearest_20 = np.argsort(distances)[:20]  # 20 nearest including self
            print(f"    Showing 20 nearest UMAP neighbors")
            
            self.fig20_segments = nearest_20.tolist()  # Already sorted by distance
            self.fig20_offset = 0
            self.fig20_selected = segment_id
            self.fig20_display_mode = 'umap_neighbors'  # Showing UMAP neighbors
            self.fig33_current_group = None
            self.fig34_current_cluster = None
            self._update_fig20()
        
        self._update_display()
    
    def _find_active_blob(self, umap_x: float, umap_y: float) -> Optional[Blob]:
        # First try: lookup by pixel (exact location on colored region)
        blob = self.detector.get_blob_at_umap_coord(umap_x, umap_y)
        if blob is not None:
            return blob
        
        # Fallback: find the nearest segment and check which blob it belongs to
        segment_id = self._find_nearest_segment(umap_x, umap_y)
        if segment_id is not None:
            # Check if this segment is in any blob
            for b in self.detector._blobs:
                # Get original segment IDs for this blob
                blob_orig_segs = self._active_indices[b.segment_indices]
                if segment_id in blob_orig_segs:
                    return b
        return None
    
    def _find_nearest_segment(self, umap_x: float, umap_y: float) -> Optional[int]:
        umap_coords = self.data.umap_coords
        distances = np.sqrt((umap_coords[:, 0] - umap_x)**2 + (umap_coords[:, 1] - umap_y)**2)
        nearest_idx = np.argmin(distances)
        return nearest_idx if distances[nearest_idx] < 0.5 else None
    
    def _on_key_press(self, event) -> None:
        key = event.key
        
        # Handle input mode (waiting for number or y/n)
        if self.input_mode is not None:
            if self.input_mode == 'finalize':
                # Handle y/n for finalize confirmation (buffered, Enter to submit)
                if key in ['y', 'Y', 'n', 'N']:
                    self.input_buffer = key
                    print(f"  Input: {key}", end='\r')
                    return
                elif key == 'Return' or key == 'enter':
                    self._process_input()
                    return
                elif key == 'escape':
                    print("\nCancelled.")
                    self.input_mode = None
                    self.input_buffer = ""
                    return
                return
            
            # For export mode, accept alphanumeric + underscore
            if self.input_mode == 'export':
                if len(key) == 1 and (key.isalnum() or key in '_-'):
                    self.input_buffer += key
                    print(f"  Filename: {self.input_buffer}", end='\r')
                    return
                elif key == 'backspace':
                    self.input_buffer = self.input_buffer[:-1]
                    print(f"  Filename: {self.input_buffer}  ", end='\r')
                    return
            elif key in '0123456789':
                self.input_buffer += key
                print(f"  Input: {self.input_buffer}", end='\r')
                return
            elif key == 'backspace':
                self.input_buffer = self.input_buffer[:-1]
                print(f"  Input: {self.input_buffer}  ", end='\r')
                return
            
            if key == 'Return' or key == 'enter':
                self._process_input()
                return
            elif key == 'escape':
                print("\nCancelled.")
                self.input_mode = None
                self.input_buffer = ""
                return
            # Ignore other keys in input mode
            return
        
        if key in ('shift', 'shift_l', 'shift_r'):
            self.shift_pressed = True
            return
        
        # Shift+C to reset (check uppercase directly - more reliable on Windows)
        if key == 'C':
            self._reset_clusters()
            return
        
        # Threshold (BLOBBING only)
        if key == '=' or key == '+':
            if self.app_mode != 'BLOBBING':
                print("Threshold adjustment only in BLOBBING mode")
                return
            self.detector.adjust_threshold(1.2)
            print(f"Threshold: {self.detector.theta:.4e}")
            self._update_blobs_threshold_only()
        elif key == '-':
            if self.app_mode != 'BLOBBING':
                print("Threshold adjustment only in BLOBBING mode")
                return
            self.detector.adjust_threshold(1.0 / 1.2)
            print(f"Threshold: {self.detector.theta:.4e}")
            self._update_blobs_threshold_only()
        
        # Radius (BLOBBING only)
        elif key == 'up':
            if self.app_mode != 'BLOBBING':
                print("Radius adjustment only in BLOBBING mode")
                return
            self.detector.adjust_radius(0.5)
            print(f"Radius: {self.detector.radius:.1f}")
            self._update_blobs_radius_only()
        elif key == 'down':
            if self.app_mode != 'BLOBBING':
                print("Radius adjustment only in BLOBBING mode")
                return
            self.detector.adjust_radius(-0.5)
            print(f"Radius: {self.detector.radius:.1f}")
            self._update_blobs_radius_only()
        
        # Navigation in Fig20 (dynamic step based on current display)
        elif key == 'right':
            if self.fig20_segments:
                # Move forward by half the current display
                step = max(3, len(self.fig20_segments) // 20)
                if self.fig20_offset + step < len(self.fig20_segments):
                    self.fig20_offset += step
                    self._update_fig20()
        elif key == 'left':
            if self.fig20_offset > 0:
                step = max(3, len(self.fig20_segments) // 20)
                self.fig20_offset = max(0, self.fig20_offset - step)
                self._update_fig20()
        
        # Page scroll for active figure (33 or 34)
        elif key == 'pagedown':
            if event.canvas == self.fig_clusters.canvas:
                n_clusters = len(set(self.data.segments['cluster_id'].values))
                if self.fig34_row_offset + self.fig34_max_rows < n_clusters:
                    self.fig34_row_offset += self.fig34_max_rows
                    self._update_cluster_visualizer()
            elif event.canvas == self.fig_blobs.canvas:
                data = self.blob_assignments if self.app_mode == 'BLOBBING' else self.precluster_assignments
                if self.fig33_row_offset + self.fig33_max_rows < len(data):
                    self.fig33_row_offset += self.fig33_max_rows
                    self._update_blob_visualizer()
        elif key == 'pageup':
            if event.canvas == self.fig_clusters.canvas:
                self.fig34_row_offset = max(0, self.fig34_row_offset - self.fig34_max_rows)
                self._update_cluster_visualizer()
            elif event.canvas == self.fig_blobs.canvas:
                self.fig33_row_offset = max(0, self.fig33_row_offset - self.fig33_max_rows)
                self._update_blob_visualizer()
        
        # Home - jump to beginning of current cluster/group in Spectrogram Viewer
        elif key == 'home':
            if self.fig20_segments:
                self.fig20_offset = 0
                self._update_fig20()
                print("Jumped to beginning")
        
        # End - jump to end of current cluster/group in Spectrogram Viewer
        elif key == 'end':
            if self.fig20_segments:
                # Calculate how many fit from the end
                AVAILABLE_WIDTH = 0.94
                MAX_COLS = int(AVAILABLE_WIDTH / SpectrogramViewer.FIXED_WIDTH_PER_COL)
                
                # Count backwards to find offset that shows last segments
                n_segs = len(self.fig20_segments)
                total_cols = 0
                offset = n_segs
                for i in range(n_segs - 1, -1, -1):
                    seg_id = self.fig20_segments[i]
                    spec = self.data.get_segment_spectrogram(seg_id)
                    cols = spec.shape[1] if spec is not None and spec.size > 0 else 0
                    if total_cols + cols > MAX_COLS:
                        break
                    total_cols += cols
                    offset = i
                
                self.fig20_offset = offset
                self._update_fig20()
                print(f"Jumped to end")
        
        # Assign blob
        elif key == 'w':
            self._assign_selected_blob()
        
        # Move segment - start input mode
        elif key == 'm':
            self._start_move_input()
        
        # Sort mode - show menu
        elif key == 'o':
            self._start_sort_input()
        
        # Merge clusters - start input mode
        elif key == 'u':
            self._start_merge_input()
        
        # Cycle modes: CLUSTERING → BLOBBING → PROOFREADING → CLUSTERING
        elif key == 'c' and not self.shift_pressed:
            self._cycle_mode()
        
        # Export to HDF5
        elif key == 'x':
            self._export_to_hdf5()
        
        # Quit
        elif key == 'q':
            plt.close('all')
    
    def _process_input(self) -> None:
        """Process the input buffer based on current input mode."""
        # Handle finalize confirmation (y/n, not numbers)
        if self.input_mode == 'finalize':
            if self.input_buffer.lower() in ['y', 'yes', '1']:
                self._do_finalize_clustering()
            else:
                print("\nFinalize cancelled.")
            self.input_mode = None
            self.input_buffer = ""
            return
        

        
        if not self.input_buffer:
            print("\nNo input provided.")
            self.input_mode = None
            return
        
        try:
            value = int(self.input_buffer)
        except ValueError:
            print(f"\nInvalid number: {self.input_buffer}")
            self.input_mode = None
            self.input_buffer = ""
            return
        
        if self.input_mode == 'move':
            seg_id = self.fig20_selected or self.fig33_selected
            if seg_id is not None:
                if self.app_mode == 'PROOFREADING':
                    self._move_segment_to_precluster(seg_id, value)
                elif self.app_mode == 'CLUSTERING':
                    self._move_segment_to_cluster(seg_id, value)
        elif self.input_mode == 'merge':
            if self.app_mode == 'PROOFREADING' and self.fig33_current_group is not None:
                self._merge_preclusters(value, self.fig33_current_group)
            elif self.app_mode == 'CLUSTERING' and self.fig34_current_cluster is not None:
                self._merge_clusters(value, self.fig34_current_cluster)
        elif self.input_mode == 'sort':
            modes = {1: 'timestamp', 2: 'duration', 3: 'random', 4: 'nn_chain', 5: 'nn_outlier'}
            if value in modes:
                if value in [4, 5]:
                    # Show sub-menu for nearest neighbor space
                    self._pending_nn_type = 'chain' if value == 4 else 'outlier'
                    print("\n" + "="*40)
                    print(f"SELECT SPACE FOR NN_{self._pending_nn_type.upper()}:")
                    print("="*40)
                    print("  1. UMAP space")
                    print("  2. PC space (100 components)")
                    print("="*40)
                    print(">>> Type number + Enter IN THE FIGURE WINDOW <<<")
                    self.input_mode = 'nearest_space'
                    self.input_buffer = ""
                    return  # Don't clear input_mode yet
                else:
                    self.fig20_sort_mode = modes[value]
                    print(f"✓ Sort mode: {self.fig20_sort_mode}")
                    self._apply_sort()
            else:
                print("Invalid choice (1-5).")
        elif self.input_mode == 'nearest_space':
            nn_type = getattr(self, '_pending_nn_type', 'chain')
            if value == 1:
                self.fig20_sort_mode = f'nn_{nn_type}_umap'
                print(f"✓ Sort mode: {self.fig20_sort_mode}")
                self._apply_sort()
            elif value == 2:
                self.fig20_sort_mode = f'nn_{nn_type}_pc'
                print(f"✓ Sort mode: {self.fig20_sort_mode}")
                self._apply_sort()
            else:
                print("Invalid choice (1-2).")
        
        self.input_mode = None
        self.input_buffer = ""
        self.input_buffer = ""
    
    def _apply_sort(self) -> None:
        """Apply the current sort mode to segments."""
        # Clear all caches so new sort mode applies everywhere
        self._fig33_sorted_cache.clear()
        self._fig34_sorted_cache.clear()
        
        if self.fig20_segments:
            # Re-sort current view
            if self.fig33_current_group is not None:
                if self.app_mode == 'BLOBBING' and self.fig33_current_group in self.blob_assignments:
                    base_segs = self.blob_assignments[self.fig33_current_group]
                elif self.fig33_current_group in self.precluster_assignments:
                    base_segs = self.precluster_assignments[self.fig33_current_group]
                else:
                    base_segs = np.array(self.fig20_segments)
                sorted_segs = self._sort_segments(base_segs)
                self._fig33_sorted_cache[self.fig33_current_group] = sorted_segs
                self.fig20_segments = sorted_segs.tolist()
            elif self.fig34_current_cluster is not None:
                cluster_segs = self.data.segments[self.data.segments['cluster_id'] == self.fig34_current_cluster].index.to_numpy()
                sorted_segs = self._sort_segments(cluster_segs)
                self._fig34_sorted_cache[self.fig34_current_cluster] = sorted_segs
                self.fig20_segments = sorted_segs.tolist()
            else:
                base_segs = np.array(self.fig20_segments)
                self.fig20_segments = self._sort_segments(base_segs).tolist()
            
            self.fig20_offset = 0
            self._update_fig20()
        
        # Always update Fig33/34 to show new sort order (caches were cleared)
        self._update_blob_visualizer()
        self._update_cluster_visualizer()
    
    def _start_move_input(self) -> None:
        """Start move segment input mode."""
        if self.app_mode not in ['PROOFREADING', 'CLUSTERING']:
            print("Move only available in PROOFREADING or CLUSTERING mode.")
            return
        
        seg_id = self.fig20_selected or self.fig33_selected
        if seg_id is None:
            print("No segment selected.")
            return
        
        if self.app_mode == 'PROOFREADING':
            # Find which precluster it's in
            old_group = None
            for pid, segs in self.precluster_assignments.items():
                if seg_id in segs.tolist():
                    old_group = pid
                    break
            prefix = "P"
        else:  # CLUSTERING
            old_group = int(self.data.segments.iloc[seg_id]['cluster_id'])
            prefix = "C"
        
        print(f"\n>>> Move segment {seg_id} ({prefix}{old_group}) to {prefix.lower()}:")
        print(">>> Type number + Enter IN THE FIGURE WINDOW <<<")
        self.input_mode = 'move'
        self.input_buffer = ""
    
    def _start_merge_input(self) -> None:
        """Start merge cluster input mode (PROOFREADING or CLUSTERING)."""
        if self.app_mode == 'BLOBBING':
            print("Merge is only available in PROOFREADING or CLUSTERING mode.")
            return
        
        if self.app_mode == 'PROOFREADING':
            if self.fig33_current_group is None:
                print("No precluster selected. Click on a precluster in the Grid first.")
                return
            available = sorted([c for c in self.precluster_assignments.keys() if c != self.fig33_current_group])
            if not available:
                print("No other preclusters to merge.")
                return
            print(f"\n>>> Merge INTO P{self.fig33_current_group}. Available: {available}")
        else:  # CLUSTERING
            if self.fig34_current_cluster is None:
                print("No cluster selected. Click on a cluster in the Grid first.")
                return
            cluster_ids = sorted(set(self.data.segments['cluster_id'].values))
            available = [c for c in cluster_ids if c != self.fig34_current_cluster]
            if not available:
                print("No other clusters to merge.")
                return
            print(f"\n>>> Merge INTO C{self.fig34_current_cluster}. Available: {available}")
        
        print(">>> Type number + Enter IN THE FIGURE WINDOW <<<")
        self.input_mode = 'merge'
        self.input_buffer = ""
    
    def _start_sort_input(self) -> None:
        """Show sort mode menu and start input mode."""
        print("\n" + "="*40)
        print("SELECT SORTING MODE (type in figure window):")
        print("="*40)
        print(f"  1. Timestamp" + (" (CURRENT)" if self.fig20_sort_mode == 'timestamp' else ""))
        print(f"  2. Duration" + (" (CURRENT)" if self.fig20_sort_mode == 'duration' else ""))
        print(f"  3. Random" + (" (CURRENT)" if self.fig20_sort_mode == 'random' else ""))
        print(f"  4. NN_chain (adjacent = similar)" + (" (CURRENT)" if 'nn_chain' in self.fig20_sort_mode else ""))
        print(f"  5. NN_outlier (typical first, outliers last)" + (" (CURRENT)" if 'nn_outlier' in self.fig20_sort_mode else ""))
        print("="*40)
        print(">>> Type number + Enter IN THE FIGURE WINDOW <<<")
        self.input_mode = 'sort'
        self.input_buffer = ""
    
    def _on_key_release(self, event) -> None:
        if event.key in ('shift', 'shift_l', 'shift_r'):
            self.shift_pressed = False
    
    def _assign_selected_blob(self) -> None:
        """Assign selected blob to new blob group (BLOBBING mode only)."""
        if self.app_mode != 'BLOBBING':
            print("Blob assignment only available in BLOBBING mode.")
            return
        if self.selected_blob is None:
            print("No blob selected.")
            return
        
        all_blob_segs = self._active_indices[self.selected_blob.segment_indices]
        # Filter out segments that were already assigned (from overlapping blob regions)
        blob_segs = np.array([s for s in all_blob_segs if s not in self.assigned_segment_indices])
        
        if len(blob_segs) == 0:
            print("All segments in this blob are already assigned.")
            self.selected_blob = None
            return
        
        self.blob_assignments[self.current_blob_id] = blob_segs.copy()
        self.assigned_segment_indices.update(blob_segs.tolist())
        
        print(f"Assigned {len(blob_segs)} segments → B{self.current_blob_id}")
        
        self.current_blob_id += 1
        self.selected_blob = None
        self.selected_segment = None
        
        # Re-detect blobs on remaining (unassigned) points
        # This now works correctly because maprange is fixed in __init__
        self._update_blobs()
        self._update_blob_visualizer()
    
    def _cycle_mode(self) -> None:
        """Cycle through modes: CLUSTERING → BLOBBING → PROOFREADING → CLUSTERING."""
        if self.app_mode == 'CLUSTERING':
            # CLUSTERING → BLOBBING
            print("\n" + "="*50)
            print("ENTERING BLOBBING MODE")
            print("="*50)
            print("  Use +/- to adjust threshold, ↑/↓ for radius")
            print("  Click blobs, press 'w' to assign")
            print("  Press 'c' when done to enter PROOFREADING")
            print("="*50 + "\n")
            
            # Clear previous assignments when re-entering BLOBBING
            self.blob_assignments.clear()
            self.precluster_assignments.clear()
            self.assigned_segment_indices.clear()
            self._fig33_sorted_cache.clear()
            self.current_blob_id = 1
            self.fig33_current_group = None
            self.fig33_row_offset = 0
            
            self.app_mode = 'BLOBBING'
            self._update_blobs()
            self._update_blob_visualizer()
            self._update_display()
        
        elif self.app_mode == 'BLOBBING':
            # BLOBBING → PROOFREADING (same as _toggle_proofread_mode)
            self._toggle_proofread_mode()
        
        elif self.app_mode == 'PROOFREADING':
            # PROOFREADING → CLUSTERING (finalize)
            self._finalize_clustering()
    
    def _toggle_proofread_mode(self) -> None:
        """Enter proofreading mode from blobbing (one-way transition)."""
        if self.app_mode != 'BLOBBING':
            print("Already in PROOFREADING/CLUSTERING mode. Cannot go back to BLOBBING.")
            return
        
        if not self.blob_assignments:
            print("No blobs assigned yet. Assign some blobs first with 'w'.")
            return
        
        print("\n" + "="*50)
        print("ENTERING PROOFREADING MODE")
        print("="*50)
        
        # Copy blob assignments to precluster assignments
        self.precluster_assignments = {k: v.copy() for k, v in self.blob_assignments.items()}
        
        # All unassigned segments go to P1
        all_indices = set(range(len(self.data.segments)))
        unassigned = all_indices - self.assigned_segment_indices
        print(f"  Total segments: {len(all_indices)}")
        print(f"  Assigned to blobs: {len(self.assigned_segment_indices)}")
        print(f"  Unassigned: {len(unassigned)}")
        
        # Unassigned segments go to P0, blobs become P1, P2... (no shifting needed)
        if unassigned:
            self.precluster_assignments[0] = np.array(list(unassigned))
            print(f"  {len(unassigned)} unassigned segments → P0")
        
        print(f"  Created {len(self.precluster_assignments)} preclusters")
        print(f"  Press 'm' to move segments, 'u' to merge preclusters")
        print("  Press 'c' to finalize clustering")
        print("="*50 + "\n")
        
        self.app_mode = 'PROOFREADING'
        self.fig33_row_offset = 0  # Reset scroll
        self._fig33_sorted_cache.clear()  # Clear cache to force re-sort with new mode
        
        # Auto-select P0 and populate Fig20
        if 0 in self.precluster_assignments:
            self.fig33_current_group = 0
            self.fig20_segments = self._sort_segments(self.precluster_assignments[0]).tolist()
            self.fig20_offset = 0
            self._update_fig20()
        
        self._update_blob_visualizer()  # Now shows P0, P1, P2,...
        self._update_display()
    
    def _finalize_clustering(self) -> None:
        """Finalize preclusters as clusters (PROOFREADING → CLUSTERING)."""
        if self.app_mode != 'PROOFREADING':
            print("Finalize only available in PROOFREADING mode.")
            return
        
        print("\n>>> Finalize clustering? (y/n) <<<")
        print(">>> Type 'y' + Enter IN THE FIGURE WINDOW <<<")
        self.input_mode = 'finalize'
        self.input_buffer = ""
    
    def _do_finalize_clustering(self) -> None:
        """Actually finalize the clustering after confirmation."""
        print("\n" + "="*50)
        print("FINALIZING CLUSTERS")
        print("="*50)
        
        # Apply precluster assignments to cluster_id
        for preclust_id, segment_indices in self.precluster_assignments.items():
            # Use iloc for positional indexing (segment_indices are integer positions)
            for idx in segment_indices:
                self.data.segments.iloc[idx, self.data.segments.columns.get_loc('cluster_id')] = preclust_id
        
        print(f"  Applied {len(self.precluster_assignments)} clusters")
        print("  Press 'x' to save to HDF5")
        print("="*50 + "\n")
        
        self.app_mode = 'CLUSTERING'
        self._fig34_sorted_cache.clear()  # Clear cluster cache
        self._update_cluster_visualizer()  # Update Cluster Grid
        self._update_display()
    
    def _move_segment_dialog(self) -> None:
        """Show dialog to move selected segment to a cluster."""
        seg_id = self.fig20_selected or self.fig33_selected
        if seg_id is None:
            print("No segment selected.")
            return
        
        old_cluster = int(self.data.segments.iloc[seg_id]['cluster_id'])
        print(f"\n>>> Segment {seg_id} is in C{old_cluster}")
        
        try:
            result = int(input("Move to cluster (enter number): "))
            self._move_segment_to_cluster(seg_id, result)
        except (ValueError, EOFError):
            print("Cancelled or invalid input.")
    
    def _move_segment_to_cluster(self, seg_id: int, target_cluster: int) -> None:
        """Move segment to target cluster (CLUSTERING mode)."""
        if self.app_mode != 'CLUSTERING':
            print("Use CLUSTERING mode for cluster moves.")
            return
        
        old_cluster = int(self.data.segments.iloc[seg_id]['cluster_id'])
        
        # Update cluster_id in the data
        self.data.segments.iloc[seg_id, self.data.segments.columns.get_loc('cluster_id')] = target_cluster
        
        print(f"Moved segment {seg_id}: C{old_cluster} → C{target_cluster}")
        
        # Clear cache for affected clusters
        if old_cluster in self._fig34_sorted_cache:
            del self._fig34_sorted_cache[old_cluster]
        if target_cluster in self._fig34_sorted_cache:
            del self._fig34_sorted_cache[target_cluster]
        
        # Update Fig 20 if showing a cluster
        if self.fig34_current_cluster is not None:
            cluster_segs = self.data.segments[self.data.segments['cluster_id'] == self.fig34_current_cluster].index.tolist()
            self.fig20_segments = self._sort_segments(np.array(cluster_segs)).tolist()
            self._update_fig20()
        
        self._update_cluster_visualizer()
    
    def _move_segment_to_precluster(self, seg_id: int, target_precluster: int) -> None:
        """Move segment to target precluster (PROOFREADING mode)."""
        if self.app_mode != 'PROOFREADING':
            print("Use PROOFREADING mode for precluster moves.")
            return
        
        # Find and remove from current precluster
        old_precluster = None
        for preclust_id, segs in list(self.precluster_assignments.items()):
            segs_list = segs.tolist() if hasattr(segs, 'tolist') else list(segs)
            if seg_id in segs_list:
                old_precluster = preclust_id
                # Remove segment from this precluster
                new_segs = np.array([s for s in segs_list if s != seg_id])
                if len(new_segs) == 0:
                    del self.precluster_assignments[preclust_id]
                else:
                    self.precluster_assignments[preclust_id] = new_segs
                break
        
        # Add to target precluster
        if target_precluster in self.precluster_assignments:
            target_segs = self.precluster_assignments[target_precluster].tolist()
            if seg_id not in target_segs:  # Avoid duplicates
                target_segs.append(seg_id)
                self.precluster_assignments[target_precluster] = np.array(target_segs)
        else:
            self.precluster_assignments[target_precluster] = np.array([seg_id])
        
        print(f"Moved segment {seg_id}: P{old_precluster} → P{target_precluster}")
        
        # Clear sorted cache for affected preclusters so they get re-sorted
        if old_precluster in self._fig33_sorted_cache:
            del self._fig33_sorted_cache[old_precluster]
        if target_precluster in self._fig33_sorted_cache:
            del self._fig33_sorted_cache[target_precluster]
        
        # Update Fig 20 if showing a precluster
        if self.fig33_current_group is not None and self.fig33_current_group in self.precluster_assignments:
            self.fig20_segments = self._sort_segments(
                self.precluster_assignments[self.fig33_current_group]).tolist()
            self._update_fig20()
        
        self._update_blob_visualizer()
    
    def _merge_preclusters(self, source_precluster: int, target_precluster: int) -> None:
        """Merge source precluster into target precluster."""
        if self.app_mode != 'PROOFREADING':
            print("Merge only available in PROOFREADING mode.")
            return
        
        if source_precluster not in self.precluster_assignments:
            print(f"Source precluster P{source_precluster} not found.")
            return
        if target_precluster not in self.precluster_assignments:
            print(f"Target precluster P{target_precluster} not found.")
            return
        
        source_segs = self.precluster_assignments[source_precluster]
        self.precluster_assignments[target_precluster] = np.append(
            self.precluster_assignments[target_precluster], source_segs)
        del self.precluster_assignments[source_precluster]
        
        print(f"Merged P{source_precluster} ({len(source_segs)} segs) into P{target_precluster}")
        
        # Clear cache for affected preclusters
        if source_precluster in self._fig33_sorted_cache:
            del self._fig33_sorted_cache[source_precluster]
        if target_precluster in self._fig33_sorted_cache:
            del self._fig33_sorted_cache[target_precluster]
        
        # Update Fig 20 if showing the target precluster
        if self.fig33_current_group == target_precluster:
            self.fig20_segments = self._sort_segments(
                self.precluster_assignments[target_precluster]).tolist()
            self._update_fig20()
        
        self._update_blob_visualizer()
    
    def _merge_clusters(self, source_cluster: int, target_cluster: int) -> None:
        """Merge source cluster into target cluster (CLUSTERING mode)."""
        if self.app_mode != 'CLUSTERING':
            print("Merge only available in CLUSTERING mode.")
            return
        
        source_cluster = int(source_cluster)
        target_cluster = int(target_cluster)
        
        # Find segments in source cluster
        source_mask = self.data.segments['cluster_id'] == source_cluster
        source_count = source_mask.sum()
        
        if source_count == 0:
            print(f"Source cluster C{source_cluster} not found or empty.")
            return
        
        # Update all source segments to target cluster
        cluster_col = self.data.segments.columns.get_loc('cluster_id')
        for idx in self.data.segments[source_mask].index:
            pos = self.data.segments.index.get_loc(idx)
            self.data.segments.iloc[pos, cluster_col] = target_cluster
        
        print(f"Merged C{source_cluster} ({source_count} segs) into C{target_cluster}")
        
        # Clear cache for affected clusters
        if source_cluster in self._fig34_sorted_cache:
            del self._fig34_sorted_cache[source_cluster]
        if target_cluster in self._fig34_sorted_cache:
            del self._fig34_sorted_cache[target_cluster]
        
        # Update Fig 20 if showing the target cluster
        if self.fig34_current_cluster == target_cluster:
            cluster_segs = self.data.segments[self.data.segments['cluster_id'] == target_cluster].index.tolist()
            self.fig20_segments = self._sort_segments(np.array(cluster_segs)).tolist()
            self._update_fig20()
        
        self._update_cluster_visualizer()
    
    def _show_sort_dialog(self) -> None:
        """Show popup dialog to select sorting mode."""
        sort_options = {
            1: ('Timestamp', 'timestamp'),
            2: ('Duration', 'duration'),
            3: ('Random', 'random'),
            4: ('Nearest Neighbor', 'nearest'),
        }
        
        print("\n" + "="*40)
        print("SELECT SORTING MODE:")
        print("="*40)
        for num, (name, mode) in sort_options.items():
            current = " (CURRENT)" if mode == self.fig20_sort_mode else ""
            print(f"  {num}. {name}{current}")
        print("="*40)
        
        try:
            choice = int(input("Enter choice (1-4): "))
            if choice in sort_options:
                _, new_mode = sort_options[choice]
                self.fig20_sort_mode = new_mode
                print(f"✓ Sort mode changed to: {new_mode}")
                
                # Re-sort current segments
                if self.fig20_segments:
                    if self.fig33_current_cluster is not None and self.fig33_current_cluster in self.assigned_blobs:
                        base_segs = self.assigned_blobs[self.fig33_current_cluster]
                    else:
                        base_segs = np.array(self.fig20_segments)
                    self.fig20_segments = self._sort_segments(base_segs).tolist()
                    self.fig20_offset = 0
                    self._update_fig20()
            else:
                print("Invalid choice.")
        except (ValueError, EOFError) as e:
            print(f"Cancelled or invalid input: {e}")
    
    def _merge_cluster_dialog(self) -> None:
        """Show dialog to merge another cluster into the current cluster."""
        if self.fig33_current_cluster is None:
            print("No cluster selected. Click on a cluster in the Cluster Grid first.")
            return
        
        target_cluster = self.fig33_current_cluster
        available_clusters = sorted([c for c in self.assigned_blobs.keys() if c != target_cluster])
        
        if not available_clusters:
            print("No other clusters to merge.")
            return
        
        print(f"\n>>> Merge INTO C{target_cluster}")
        print(f"Available clusters: {available_clusters}")
        
        try:
            result = int(input("Merge which cluster? "))
            if result not in self.assigned_blobs:
                print(f"Cluster C{result} does not exist.")
                return
            if result == target_cluster:
                print("Cannot merge cluster into itself.")
                return
            self._merge_clusters(result, target_cluster)
        except (ValueError, EOFError):
            print("Cancelled or invalid input.")
    
    def _reset_clusters(self) -> None:
        """Reset all assignments and return to BLOBBING mode."""
        self.data.reset_clusters()
        self.blob_assignments.clear()
        self.precluster_assignments.clear()
        self.assigned_segment_indices.clear()
        self._fig33_sorted_cache.clear()
        self._fig34_sorted_cache.clear()
        self.current_blob_id = 1
        self.selected_blob = None
        self.fig20_segments = []
        self.fig20_selected = None
        self.fig33_selected = None
        self.fig33_current_group = None
        self.fig33_row_offset = 0
        self.app_mode = 'BLOBBING'
        self._update_blobs()
        self._update_blob_visualizer()
        self._update_display()
        print("Reset all")
    

    
    def _export_to_hdf5(self) -> None:
        """Start export to HDF5 using native save dialog."""
        print("\n" + "="*60)
        print("EXPORT TO HDF5")
        print("="*60)
        
        # Default filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"clusters_{timestamp}.h5"
        
        # Use tkinter for save dialog
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
                initialdir=Path.cwd(),
                initialfile=default_name,
                title="Export Clusters to HDF5"
            )
            
            root.destroy()
            
            if file_path:
                self._do_export(file_path)
            else:
                print("Export cancelled.")
        except Exception as e:
            print(f"Dialog error: {e}")
            print("Falling back to auto-name export...")
            self._do_export(default_name)
    
    def _do_export(self, filename: str) -> None:
        """Actually perform the export."""
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clusters_{timestamp}.h5"
        
        if not filename.endswith('.h5'):
            filename += '.h5'
        
        output_path = Path.cwd() / filename
        print(f"Exporting to: {output_path}")
        
        try:
            save_to_hdf5(self.data, str(output_path), compress=True)
            print(f"✓ Export complete: {output_path}")
        except Exception as e:
            print(f"Export error: {e}")
    
    def run(self) -> None:
        print("\n" + "="*60)
        print("UMAP Clustering Tool")
        print("="*60)
        print("  =/- : Threshold    ↑/↓ : Radius    ←/→ : Navigate")
        print("  w   : Assign blob  m   : Move segment")
        print("  u   : Merge        o   : Sort mode")
        print("  c   : Cycle modes  x : Export (HDF5)")
        print("  q   : Quit")
        print("="*60 + "\n")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Semi-Automated Clustering Tool")
    parser.add_argument("data_path", help="Path to the HDF5 data file")
    args = parser.parse_args()
    
    data_path = args.data_path
    if not Path(data_path).exists():
        print(f"Error: File not found: {data_path}")
        return

    data = load_data(data_path)
    print(f"Loaded {data.n_segments} segments from {data.n_files} files")
    
    ClusteringApp(data).run()


if __name__ == "__main__":
    main()

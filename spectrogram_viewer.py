"""
Spectrogram viewer component for displaying multiple spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

try:
    from .data_loader import FlatData
except ImportError:
    from data_loader import FlatData


class SpectrogramViewer:
    """Displays spectrograms with consistent width scaling."""
    
    # Fixed width per spectrogram column - this constant ensures consistent scaling
    # across all views (Spectrogram Viewer, Blob/Precluster Grid, Cluster Grid)
    FIXED_WIDTH_PER_COL = 0.002
    
    def __init__(self, data: FlatData, fig: plt.Figure, n_rows: int = 1, n_cols: int = 12):
        self.data = data
        self.fig = fig
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.context_sec = 0.0
        self._segment_axes = {}  # Map from axis id to segment id

    def set_context_seconds(self, context_sec: float) -> None:
        """Kept for API compatibility but Spectrogram Viewer always renders without context."""
        pass  # Context is now handled exclusively by the Context Viewer
    
    def display_with_highlight(self, segment_ids: List[int], highlight_id: Optional[int] = None) -> None:
        """Display spectrograms with optional highlight. Always renders without context."""
        self.fig.clear()
        self._segment_axes = {}
        
        if not segment_ids:
            return
        
        # Get spectrogram dimensions for each segment (always without context)
        specs = []
        col_counts = []
        for seg_id in segment_ids:
            spec = self.data.get_segment_spectrogram(seg_id, context_sec=0.0)
            if spec is not None and spec.size > 0:
                specs.append(spec)
                col_counts.append(spec.shape[1])
            else:
                specs.append(None)
                col_counts.append(0)
        
        total_cols = sum(col_counts)
        if total_cols == 0:
            return
        
        # Layout parameters
        left_margin = 0.02
        right_margin = 0.02
        top_margin = 0.12  # Space for title
        bottom_margin = 0.08
        gap = 0.003
        
        available_width = 1.0 - left_margin - right_margin
        available_height = 1.0 - top_margin - bottom_margin
        
        # Calculate positions based on fixed width per column
        x_pos = left_margin
        row_height = available_height / self.n_rows
        
        for idx, (seg_id, spec, n_cols) in enumerate(zip(segment_ids, specs, col_counts)):
            if spec is None or n_cols == 0:
                continue
            
            # Width based on number of columns with fixed scaling
            width = n_cols * self.FIXED_WIDTH_PER_COL
            
            # Check if we've exceeded available width
            if x_pos + width > 1.0 - right_margin:
                break
            
            # Create axis for this spectrogram
            axis_width = max(width - gap, 0.001)
            ax = self.fig.add_axes([x_pos, bottom_margin, axis_width, row_height])
            
            # Display spectrogram
            ax.imshow(spec[::-1, :], aspect='auto', cmap='inferno', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Highlight if selected
            if seg_id == highlight_id:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            
            # Store mapping from axis to segment
            self._segment_axes[id(ax)] = seg_id
            
            x_pos += width + gap
        
        self.fig.canvas.draw_idle()
    
    def get_clicked_segment(self, event) -> Optional[int]:
        """Get segment ID from click event."""
        if event.inaxes is None:
            return None
        
        ax_id = id(event.inaxes)
        return self._segment_axes.get(ax_id)

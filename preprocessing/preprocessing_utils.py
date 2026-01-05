"""
Utility functions for audio preprocessing pipeline.
"""
import os
import re
import numpy as np
from glob import glob


def convert_day_to_str(day):
    """Convert a day number to a zero-padded 3-digit string if applicable.
    
    Args:
        day: Integer or string representing the day number or subdir name.
        
    Returns:
        Zero-padded string (e.g., 1 -> '001') or original string if not numeric or too long.
    """
    day_str = str(day)
    # Only pad if it looks like a small integer (up to 3 digits)
    if day_str.isdigit() and len(day_str) <= 3:
        return day_str.zfill(3)
    return day_str


def get_all_days(data_dir, max_day=900):
    """Get all subdirectories that contain wav files.
    
    Args:
        data_dir: Path to directory containing subfolders.
        max_day: Deprecated, kept for API compatibility.
        
    Returns:
        Sorted list of subdirectory names (as strings) containing .wav files.
        Returns [None] if there are no matchin subfolders but wav files exist in flat structure.
    """
    valid_groups = []
    
    # Check all subdirectories
    try:
        subdirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    except FileNotFoundError:
        return []

    for d in subdirs:
        # Check if subdir contains wav files (shallow check)
        if glob(os.path.join(data_dir, d, '*.wav')):
            valid_groups.append(d)
    
    # If no day folders found, check if there are wav files directly in data_dir
    if not valid_groups:
        wav_files = glob(os.path.join(data_dir, '*.wav'))
        if wav_files:
            # Return None to indicate flat structure
            return [None]
    
    return sorted(valid_groups)


def get_wav_files_for_day(data_dir, day):
    """Get wav files for a specific day or for flat folder structure.
    
    Args:
        data_dir: Path to data directory.
        day: Subdir name (str) or None for flat structure.
        
    Returns:
        List of wav file paths.
    """
    if day is None:
        # Flat structure - wav files directly in data_dir
        return glob(os.path.join(data_dir, '*.wav'))
    else:
        # Day/Subdir folder structure
        # Use day directly as it is now likely a string from get_all_days
        # But we still apply convert for backward compatibility if an int was passed
        day_str = convert_day_to_str(day) 
        
        # If the direct join works, use it (handles non-numeric folders)
        path_direct = os.path.join(data_dir, str(day))
        
        # If not, try the padded version (handles "1" -> "001" logic if that was implicit)
        # But wait, get_all_days returns actual directory names now.
        # So we should just use str(day).
        return glob(os.path.join(data_dir, str(day), '*.wav'))


def get_day_string(day):
    """Get day string for file naming.
    
    Args:
        day: Subdir name (str) or None for flat structure.
        
    Returns:
        Day string (e.g., '045', 'subfolder') or 'all' for flat structure.
    """
    if day is None:
        return 'all'
    return convert_day_to_str(day)


def squeeze_one_dim_if_exists(arr: np.ndarray, axis=0):
    """Squeeze array along axis if that dimension is 1.
    
    Args:
        arr: NumPy array.
        axis: Axis to squeeze.
        
    Returns:
        Squeezed array if axis dimension was 1, otherwise original array.
    """
    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr

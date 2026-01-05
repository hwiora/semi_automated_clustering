"""
Step 2: Compute spectrograms from audio files.
"""
import os
import pickle
from glob import glob

import numpy as np
import librosa
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_wav_files_for_day, get_day_string


def spec_to_int8(spec_in_float):
    """Convert spectrogram from float to int8 for storage efficiency.
    
    Args:
        spec_in_float: Spectrogram array in float format.
        
    Returns:
        Spectrogram normalized and converted to int8.
    """
    normalized_spec = spec_in_float / np.max(np.abs(spec_in_float))
    spec_in_int8 = (normalized_spec * 127).astype(np.int8)
    return spec_in_int8


def compute_spectrograms(subject_name, data_dir, output_dir, 
                          sr=32000, n_fft=512, hop_length=128,
                          min_freq=312, max_freq=8000):
    """Compute spectrograms for all audio files.
    
    Args:
        subject_name: Identifier for the subject (used in output filenames).
        data_dir: Directory containing wav files (either directly or in day folders).
        output_dir: Directory to save spectrogram files.
        sr: Sample rate in Hz.
        n_fft: FFT size in samples.
        hop_length: Hop length in samples.
        min_freq: Minimum frequency to include in Hz.
        max_freq: Maximum frequency to include in Hz.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_days = get_all_days(data_dir)
    
    print('Computing spectrograms...')
    for day in tqdm(all_days):
        wav_fnames = []
        spectrograms = []
        timepoints_spec = []
        frequencies_spec = []
        
        day_str = get_day_string(day)
        save_path = os.path.join(output_dir, f'{subject_name}_{day_str}_subdir_spectrograms.pkl')
        
        if os.path.exists(save_path):
            print(f'{save_path} already exists, skipping')
            continue
        
        wav_files = get_wav_files_for_day(data_dir, day)
        for wav_path in tqdm(wav_files, desc=f'Subdirectory {day_str}', leave=False):
            try:
                y, _ = librosa.load(wav_path, sr=sr)
            except Exception as e:
                print(f'File loading error - {wav_path}: {e}')
                continue
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), 
                ref=np.max
            )
            D = spec_to_int8(D)
            
            t = np.arange(D.shape[1]) * hop_length / sr
            f = np.arange(D.shape[0]) * sr / n_fft
            f_window = np.logical_and(min_freq < f, f < max_freq)
            D = D[f_window, :]
            
            timepoints_spec.append(t)
            frequencies_spec.append(f)
            spectrograms.append(D)
            wav_fnames.append(wav_path)
        
        if len(spectrograms) == 0:
            print(f'No wav files found for day {day_str}')
            continue
            
        spec_dict = {
            'filename': wav_fnames,
            'timepoints': timepoints_spec,
            'frequencies': frequencies_spec,
            'spectrogram': spectrograms
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(spec_dict, f)

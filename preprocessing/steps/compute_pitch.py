"""
Step 3: Extract pitch from audio files using Parselmouth (Praat).
"""
import os
import pickle
from glob import glob

import parselmouth
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_wav_files_for_day, get_day_string


def compute_pitch(subject_name, data_dir, output_dir, sr=32000, pitch_floor=312):
    """Extract pitch contours from all audio files.
    
    Args:
        subject_name: Identifier for the subject (used in output filenames).
        data_dir: Directory containing wav files (either directly or in day folders).
        output_dir: Directory to save pitch files.
        sr: Sample rate in Hz.
        pitch_floor: Minimum pitch frequency in Hz.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_days = get_all_days(data_dir)
    
    print('Computing pitch...')
    for day in tqdm(all_days):
        wav_fnames = []
        timepoints = []
        pitch_vals = []
        
        day_str = get_day_string(day)
        save_path = os.path.join(output_dir, f'{subject_name}_{day_str}_subdir_pitch.pkl')
        
        if os.path.exists(save_path):
            print(f'{save_path} already exists, skipping')
            continue
        
        wav_files = get_wav_files_for_day(data_dir, day)
        for wav_path in tqdm(wav_files, desc=f'Subdirectory {day_str}', leave=False):
            try:
                snd = parselmouth.Sound(wav_path)
            except Exception as e:
                print(f'File loading error - {wav_path}: {e}')
                continue
            
            pitch = snd.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=sr // 4)
            pitch_vals.append(pitch.selected_array['frequency'])
            timepoints.append(pitch.xs())
            wav_fnames.append(os.path.basename(wav_path))
        
        if len(pitch_vals) == 0:
            print(f'No wav files found for day {day_str}')
            continue
            
        pitch_dict = {
            'filename': wav_fnames,
            'timepoints': timepoints,
            'pitch': pitch_vals
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(pitch_dict, f)

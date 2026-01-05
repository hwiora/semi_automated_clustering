"""
Step 4: Run WhisperSeg for audio segmentation.
"""
import os
import pickle
from glob import glob

import librosa
from tqdm import tqdm

# Import from your forked WhisperSeg
from whisperseg.model import WhisperSegmenterFast

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_wav_files_for_day, get_day_string


def run_whisperseg(subject_name, data_dir, output_dir, 
                   model_path="nccratliri/whisperseg-base-animal-vad-ct2",
                   sr=32000, min_freq=312, spec_time_step=0.002,
                   min_segment_length=0.01, eps=0.02, num_trials=3,
                   device="cuda"):
    """Run WhisperSeg segmentation on all audio files.
    
    Args:
        subject_name: Identifier for the subject (used in output filenames).
        data_dir: Directory containing wav files (either directly or in day folders).
        output_dir: Directory to save segmentation files.
        model_path: HuggingFace model path or local path for WhisperSeg.
        sr: Sample rate in Hz.
        min_freq: Minimum frequency for segmentation.
        spec_time_step: Spectrogram time step in seconds.
        min_segment_length: Minimum segment length in seconds.
        eps: DBSCAN epsilon for segment consolidation.
        num_trials: Number of segmentation trials for voting.
        device: Device to run on ('cuda' or 'cpu').
    """
    os.makedirs(output_dir, exist_ok=True)
    all_days = get_all_days(data_dir)
    
    segmenter = WhisperSegmenterFast(model_path=model_path, device=device)
    
    print('Running WhisperSeg segmentation...')
    for day in tqdm(all_days):
        wav_fnames = []
        seg_onsets = []
        seg_offsets = []
        
        day_str = get_day_string(day)
        save_path = os.path.join(output_dir, f'{subject_name}_{day_str}_subdir_segmentations.pkl')
        
        if os.path.exists(save_path):
            print(f'{save_path} already exists, skipping')
            continue
        
        wav_files = get_wav_files_for_day(data_dir, day)
        for wav_path in tqdm(wav_files, desc=f'Subdirectory {day_str}', leave=False):
            wav_path = wav_path.replace('\\', '/')
            try:
                y, _ = librosa.load(wav_path, sr=sr)
            except Exception as e:
                print(f'File loading error - {wav_path}: {e}')
                continue
            
            seg_pred = segmenter.segment(
                y, sr=sr,
                min_frequency=min_freq,
                spec_time_step=spec_time_step,
                min_segment_length=min_segment_length,
                eps=eps,
                num_trials=num_trials
            )
            
            wav_fnames.append(os.path.basename(wav_path))
            seg_onsets.append(seg_pred['onset'])
            seg_offsets.append(seg_pred['offset'])
        
        if len(wav_fnames) == 0:
            print(f'No wav files found for day {day_str}')
            continue
            
        seg_dict = {
            'filename': wav_fnames,
            'onsets': seg_onsets,
            'offsets': seg_offsets
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(seg_dict, f)

"""
Step 5: Compute embeddings from segmented audio.
"""
import os
import pickle
from glob import glob

import numpy as np
import librosa
import ctranslate2
import torch
from tqdm import tqdm

# Import from your forked WhisperSeg
from whisperseg.model4embedding import WhisperSegmenterFast as EmbedderModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_wav_files_for_day, get_day_string


def compute_embeddings(subject_name, data_dir, segmentation_dir, output_dir,
                       model_path='Systran/faster-whisper-large-v3',
                       sr=32000, min_freq=0, spec_time_step=0.0025, num_trials=3,
                       embedding_style='first_timebin', embedding_dim=1280,
                       context_window=0, batch_size=32, device="cuda"):
    """Compute embeddings for all segmented vocalizations.
    
    Args:
        subject_name: Identifier for the subject (used in output filenames).
        data_dir: Directory containing wav files (either directly or in day folders).
        segmentation_dir: Directory containing segmentation pickle files.
        output_dir: Directory to save embedding files.
        model_path: HuggingFace model path for embedding extraction.
        sr: Sample rate in Hz.
        min_freq: Minimum frequency for feature extraction.
        spec_time_step: Spectrogram time step in seconds.
        num_trials: Number of trials for feature extraction.
        embedding_style: Style of embedding extraction (e.g., 'first_timebin').
        embedding_dim: Dimension of embeddings.
        context_window: Context window in milliseconds around each segment.
        batch_size: Batch size for processing.
        device: Device to run on ('cuda' or 'cpu').
    """
    os.makedirs(os.path.join(output_dir, embedding_style), exist_ok=True)
    all_days = get_all_days(data_dir)
    
    embedder = EmbedderModel(model_path, device=device)
    
    print('Computing embeddings...')
    for day in tqdm(all_days):
        day_str = get_day_string(day)
        
        save_path = os.path.join(output_dir, embedding_style, 
                                  f'{subject_name}_{day_str}_subdir_embeddings.pkl')
        if os.path.exists(save_path):
            print(f'{save_path} already exists, skipping')
            continue
        
        # Load segmentation
        seg_path = os.path.join(segmentation_dir, 
                                f'{subject_name}_{day_str}_subdir_segmentations.pkl')
        if not os.path.exists(seg_path):
            print(f'Segmentation not found: {seg_path}, skipping')
            continue
            
        seg_dict = pickle.load(open(seg_path, 'rb'))
        
        # Collect all segments
        fname_list = []
        seg_range_list = []
        for fname, onsets, offsets in zip(seg_dict['filename'], 
                                           seg_dict['onsets'], 
                                           seg_dict['offsets']):
            for onset, offset in zip(onsets, offsets):
                fname_list.append(fname)
                seg_range_list.append([onset, offset])
        
        n_vocs = len(seg_range_list)
        if n_vocs == 0:
            print(f'No segments found for day {day_str}, skipping')
            continue
            
        emb_mat = np.zeros((n_vocs, embedding_dim), dtype=np.float32)
        
        # Get the base directory for wav files
        if day is None:
            wav_base_dir = data_dir
        else:
            wav_base_dir = os.path.join(data_dir, get_day_string(day))
        
        # Process in batches
        batch_audio_segments = []
        batch_indices = []
        fname_prev = ''
        
        for idx, (fname, seg_range) in tqdm(enumerate(zip(fname_list, seg_range_list)), 
                                            total=len(fname_list), 
                                            desc=f'Subdirectory {day_str}', 
                                            leave=False):
            if fname != fname_prev:
                wav_path = os.path.join(wav_base_dir, fname)
                try:
                    audio, _ = librosa.load(wav_path, sr=sr)
                except Exception as e:
                    print(f'File loading error - {wav_path}: {e}')
                    continue
                file_timepoints = np.arange(audio.shape[0]) / sr
                fname_prev = fname
            
            onset, offset = seg_range
            segment_window = np.logical_and(
                file_timepoints > onset - context_window / 1000,
                file_timepoints < offset + context_window / 1000
            )
            audio_segment = audio[segment_window]
            batch_audio_segments.append(audio_segment)
            batch_indices.append(idx)
            
            # Process batch
            if len(batch_audio_segments) >= batch_size or idx == len(fname_list) - 1:
                ftrs = [embedder.get_sliced_audios_features(
                    seg, sr, min_freq, spec_time_step, num_trials)[0][2]
                    for seg in batch_audio_segments]
                features = ctranslate2.StorageView.from_array(np.asarray(ftrs))
                encoded = embedder.model_list[0].encode(features)
                embeddings = torch.tensor(np.array(encoded).tolist(), device="cpu")
                embeddings = embeddings.squeeze(1).permute(0, 2, 1)  # [batch, 1280, 500]
                
                for i, emb_idx in enumerate(batch_indices):
                    emb_mat[emb_idx, :] = embeddings[i, :, 0].numpy()
                
                batch_audio_segments = []
                batch_indices = []
        
        emb_dict = {
            'filename': fname_list,
            'segment_range': seg_range_list,
            'embedding': emb_mat
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(emb_dict, f)

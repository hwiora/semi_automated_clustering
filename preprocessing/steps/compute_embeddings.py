"""
Step 5: Compute embeddings from segmented audio.

Provides both sequential (compute_embedding) and batch (compute_embedding_batch)
versions, matching the external reference implementation.
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


def compute_embedding(subject_name, data_dir, segmentation_dir, output_dir,
                      model_path='Systran/faster-whisper-large-v3',
                      sr=32000, min_freq=0, spec_time_step=0.0025, num_trials=3,
                      embedding_style='first_timebin', embedding_dim=1280,
                      context_window=0, device="cuda"):
    """Compute embeddings one segment at a time (sequential).

    Follows the external reference implementation (preprocess5_embeddings.compute_embedding).

    Args:
        subject_name: Identifier for the subject.
        data_dir: Directory containing wav files (directly or in day folders).
        segmentation_dir: Directory containing segmentation pickle files.
        output_dir: Directory to save embedding files.
        model_path: HuggingFace model path for embedding extraction.
        sr: Sample rate in Hz.
        min_freq: Minimum frequency for feature extraction.
        spec_time_step: Spectrogram time step in seconds.
        num_trials: Number of trials for feature extraction.
        embedding_style: Style of embedding extraction (e.g. 'first_timebin').
        embedding_dim: Dimension of embeddings.
        context_window: Context window in milliseconds around each segment.
        device: Device to run on ('cuda' or 'cpu').
    """
    os.makedirs(os.path.join(output_dir, embedding_style), exist_ok=True)
    all_days = get_all_days(data_dir)

    embedder = EmbedderModel(model_path, device=device)

    print('Computing Embeddings...')
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

        # Collect all valid segments
        fname_list = []
        seg_range_list = []
        n_invalid_segments = 0
        for fname, onsets, offsets in zip(seg_dict['filename'],
                                          seg_dict['onsets'],
                                          seg_dict['offsets']):
            for onset, offset in zip(onsets, offsets):
                try:
                    onset_f = float(onset)
                    offset_f = float(offset)
                except Exception:
                    n_invalid_segments += 1
                    continue
                if (not np.isfinite(onset_f)) or (not np.isfinite(offset_f)) or (offset_f <= onset_f):
                    n_invalid_segments += 1
                    continue
                fname_list.append(fname)
                seg_range_list.append([onset_f, offset_f])

        n_vocs = len(seg_range_list)
        if n_vocs == 0:
            print(f'No segments found for day {day_str}, skipping')
            continue
        if n_invalid_segments > 0:
            print(f'  Day {day_str}: skipped {n_invalid_segments} invalid segments before embedding')

        emb_mat = np.zeros((n_vocs, embedding_dim), dtype=np.float32)

        # Get wav base directory
        if day is None:
            wav_base_dir = data_dir
        else:
            wav_base_dir = os.path.join(data_dir, str(day))

        cnt = 0
        fname_prev = ''
        for fname, seg_range in zip(fname_list, seg_range_list):
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

            # Compute embedding for single segment
            ftr = embedder.get_sliced_audios_features(
                audio_segment, sr, min_freq, spec_time_step, num_trials)
            mel_spec = ftr[0][2]
            features = ctranslate2.StorageView.from_array(np.asarray([mel_spec]))
            encoded = embedder.model_list[0].encode(features)
            embedding = torch.tensor(np.array(encoded).tolist(), device="cpu")
            embedding = embedding.squeeze(0).T       # [500, 1280] -> [1280, 500]
            emb_first_timebin = embedding[:, 0]      # [1280]

            emb_mat[cnt, :] = emb_first_timebin.numpy()
            cnt += 1

        # Trim to actual count (in case some files failed to load)
        emb_mat = emb_mat[:cnt]
        fname_list = fname_list[:cnt]
        seg_range_list = seg_range_list[:cnt]

        emb_dict = {
            'filename': fname_list,
            'segment_range': seg_range_list,
            'embedding': emb_mat
        }

        with open(save_path, 'wb') as f:
            pickle.dump(emb_dict, f)


def compute_embedding_batch(subject_name, data_dir, segmentation_dir, output_dir,
                            model_path='Systran/faster-whisper-large-v3',
                            sr=32000, min_freq=0, spec_time_step=0.0025, num_trials=3,
                            embedding_style='first_timebin', embedding_dim=1280,
                            context_window=0, batch_size=32, device="cuda"):
    """Compute embeddings in batches (faster on GPU).

    Follows the external reference implementation (preprocess5_embeddings.compute_embedding_batch).

    Args:
        subject_name: Identifier for the subject.
        data_dir: Directory containing wav files (directly or in day folders).
        segmentation_dir: Directory containing segmentation pickle files.
        output_dir: Directory to save embedding files.
        model_path: HuggingFace model path for embedding extraction.
        sr: Sample rate in Hz.
        min_freq: Minimum frequency for feature extraction.
        spec_time_step: Spectrogram time step in seconds.
        num_trials: Number of trials for feature extraction.
        embedding_style: Style of embedding extraction (e.g. 'first_timebin').
        embedding_dim: Dimension of embeddings.
        context_window: Context window in milliseconds around each segment.
        batch_size: Batch size for processing.
        device: Device to run on ('cuda' or 'cpu').
    """
    os.makedirs(os.path.join(output_dir, embedding_style), exist_ok=True)
    all_days = get_all_days(data_dir)

    embedder = EmbedderModel(model_path, device=device)

    print('Computing Embeddings...')
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

        # Collect all valid segments
        fname_list = []
        seg_range_list = []
        n_invalid_segments = 0
        for fname, onsets, offsets in zip(seg_dict['filename'],
                                          seg_dict['onsets'],
                                          seg_dict['offsets']):
            for onset, offset in zip(onsets, offsets):
                try:
                    onset_f = float(onset)
                    offset_f = float(offset)
                except Exception:
                    n_invalid_segments += 1
                    continue
                if (not np.isfinite(onset_f)) or (not np.isfinite(offset_f)) or (offset_f <= onset_f):
                    n_invalid_segments += 1
                    continue
                fname_list.append(fname)
                seg_range_list.append([onset_f, offset_f])

        n_vocs = len(seg_range_list)
        if n_vocs == 0:
            print(f'No segments found for day {day_str}, skipping')
            continue
        if n_invalid_segments > 0:
            print(f'  Day {day_str}: skipped {n_invalid_segments} invalid segments before embedding')

        emb_mat = np.zeros((n_vocs, embedding_dim), dtype=np.float32)

        # Get wav base directory
        if day is None:
            wav_base_dir = data_dir
        else:
            wav_base_dir = os.path.join(data_dir, str(day))

        batch_audio_segments = []
        batch_indices = []
        fname_prev = ''

        for idx, (fname, seg_range) in enumerate(zip(fname_list, seg_range_list)):
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


# Default entry point: use batch version for speed
compute_embeddings = compute_embedding_batch


def compute_embedding_for_segments(audio, sr, onsets, offsets,
                                   model_path='Systran/faster-whisper-large-v3',
                                   min_freq=0, spec_time_step=0.0025, num_trials=3,
                                   embedding_dim=1280, context_window=0,
                                   batch_size=32, device='cuda'):
    """Compute embeddings directly from audio and segment times.

    Standalone version — no file I/O, no day/subject structure required.

    Args:
        audio: np.ndarray of audio samples (1-D, already loaded at ``sr``).
        sr: Sample rate in Hz.
        onsets: List/array of segment onset times in seconds.
        offsets: List/array of segment offset times in seconds.
        model_path: HuggingFace model path for embedding extraction.
        min_freq: Minimum frequency for feature extraction.
        spec_time_step: Spectrogram time step in seconds.
        num_trials: Number of trials for feature extraction.
        embedding_dim: Dimension of each embedding vector.
        context_window: Extra context in milliseconds added around each segment.
        batch_size: Number of segments processed per GPU batch.
        device: Device to run on ('cuda' or 'cpu').

    Returns:
        np.ndarray of shape (n_segments, embedding_dim).
    """
    onsets = list(onsets)
    offsets = list(offsets)
    if len(onsets) != len(offsets):
        raise ValueError('onsets and offsets must have the same length')

    embedder = EmbedderModel(model_path, device=device)
    file_timepoints = np.arange(len(audio)) / sr

    n_vocs = len(onsets)
    emb_mat = np.zeros((n_vocs, embedding_dim), dtype=np.float32)

    batch_audio_segments = []
    batch_indices = []

    for idx, (onset, offset) in enumerate(tqdm(zip(onsets, offsets), total=n_vocs, desc='Embedding segments')):
        onset, offset = float(onset), float(offset)
        segment_window = np.logical_and(
            file_timepoints > onset - context_window / 1000,
            file_timepoints < offset + context_window / 1000,
        )
        batch_audio_segments.append(audio[segment_window])
        batch_indices.append(idx)

        if len(batch_audio_segments) >= batch_size or idx == n_vocs - 1:
            ftrs = [embedder.get_sliced_audios_features(
                seg, sr, min_freq, spec_time_step, num_trials)[0][2]
                for seg in batch_audio_segments]
            features = ctranslate2.StorageView.from_array(np.asarray(ftrs))
            encoded = embedder.model_list[0].encode(features)
            embeddings = torch.tensor(np.array(encoded).tolist(), device='cpu')
            embeddings = embeddings.squeeze(1).permute(0, 2, 1)  # [batch, 1280, 500]

            for i, emb_idx in enumerate(batch_indices):
                emb_mat[emb_idx, :] = embeddings[i, :, 0].numpy()

            batch_audio_segments = []
            batch_indices = []

    return emb_mat

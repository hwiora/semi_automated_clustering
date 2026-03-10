"""
Step: Export preprocessing results to HDF5 format for clustering_app.

This is the bridge between the per-day pickle files produced by the preprocessing
pipeline and the single HDF5 file consumed by clustering_app.py.

Data flow:
    WhisperSeg segmentations  →  onset/offset per segment
    Whisper encoder embeddings →  1280-d vector per segment
    UMAP(embeddings)           →  2-d coordinates per segment
    PCA(embeddings)            →  100-d coordinates per segment (auxiliary)
    Spectrograms               →  full spectrogram per wav file (int8)
    Pitch                      →  continuous f0 per wav file (float32)

All arrays are aligned by the ordering that comes out of the embedding pickles
(which iterate over days → files → segments in the same order as segmentation).
"""
import os
import pickle
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import librosa
from tqdm import tqdm
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_utils import get_all_days, get_day_string
from preprocessing_utils import get_wav_files_for_day


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_umap_alignment(all_filenames, all_onset_sec, umap_dict, n_embeddings):
    """Validate that the UMAP pickle is aligned with the embedding data.

    Returns (umap_coords, list_of_warnings).
    Raises ValueError when the mismatch is unrecoverable.
    """
    umap_coords = np.asarray(umap_dict.get('umap_coordinates', []), dtype=np.float32)
    warnings = []

    # --- shape check ---
    if umap_coords.ndim != 2 or umap_coords.shape[1] < 2:
        raise ValueError(
            f'UMAP has invalid shape {umap_coords.shape}; expected (n_segments, 2). '
            'Re-run --compute-umap.'
        )

    if umap_coords.shape[0] != n_embeddings:
        raise ValueError(
            f'UMAP segment count ({umap_coords.shape[0]}) != embedding count '
            f'({n_embeddings}). Re-run --compute-umap after --compute-embeddings '
            'so they stay in sync.'
        )

    # --- filename / onset order check (tolerant) ---
    umap_fnames = umap_dict.get('filename')
    umap_ranges = umap_dict.get('segment_range')

    if umap_fnames is not None and umap_ranges is not None:
        if len(umap_fnames) != len(all_filenames):
            raise ValueError(
                f'UMAP filename list length ({len(umap_fnames)}) != embedding '
                f'filename list length ({len(all_filenames)}). Re-run --compute-umap.'
            )

        for idx in range(len(all_filenames)):
            fn_emb = all_filenames[idx]
            fn_umap = umap_fnames[idx]
            if fn_emb != fn_umap:
                raise ValueError(
                    f'UMAP/embedding filename mismatch at segment {idx}: '
                    f'{fn_emb!r} vs {fn_umap!r}. Re-run --compute-umap.'
                )

            onset_emb = all_onset_sec[idx]
            onset_umap = float(umap_ranges[idx][0])
            # Use generous tolerance — float32 pickle round-trips can lose precision
            if not np.isclose(onset_emb, onset_umap, atol=1e-3):
                warnings.append(
                    f'Onset mismatch at segment {idx}: emb={onset_emb:.6f} '
                    f'vs umap={onset_umap:.6f} (atol=1e-3). '
                    'Minor float32 drift — proceeding anyway.'
                )
                if len(warnings) >= 5:
                    warnings.append('... (suppressing further onset warnings)')
                    break

    return umap_coords[:, :2], warnings


def _subset_umap_by_segment_ranges(all_filenames, all_segment_ranges, umap_dict):
    """Subset a full UMAP table to match a subset of embeddings by segment identity.

    Identity key is (basename(filename), onset, offset) with rounded float keys
    to tolerate pickle float32/float64 drift.
    """
    umap_coords = np.asarray(umap_dict.get('umap_coordinates', []), dtype=np.float32)
    umap_fnames = umap_dict.get('filename')
    umap_ranges = umap_dict.get('segment_range')

    if umap_fnames is None or umap_ranges is None:
        raise ValueError(
            'UMAP subset extraction requires filename and segment_range fields. '
            'Re-run --compute-umap from the same embedding source.'
        )

    if len(umap_fnames) != len(umap_ranges) or len(umap_fnames) != len(umap_coords):
        raise ValueError(
            'UMAP fields length mismatch: filename / segment_range / umap_coordinates '
            'must all have the same length.'
        )

    buckets = defaultdict(list)
    for idx, (fname, seg_range) in enumerate(zip(umap_fnames, umap_ranges)):
        try:
            onset = float(seg_range[0])
            offset = float(seg_range[1])
        except Exception:
            continue
        key = (os.path.basename(str(fname)), round(onset, 3), round(offset, 3))
        buckets[key].append(idx)

    selected_indices = []
    missing = []
    for fname, seg_range in zip(all_filenames, all_segment_ranges):
        onset = float(seg_range[0])
        offset = float(seg_range[1])
        key = (os.path.basename(str(fname)), round(onset, 3), round(offset, 3))
        idx_list = buckets.get(key)
        if idx_list:
            selected_indices.append(idx_list.pop())
        else:
            if len(missing) < 10:
                missing.append((fname, onset, offset))

    if len(selected_indices) != len(all_filenames):
        raise ValueError(
            f'Unable to subset UMAP to embedding subset: matched {len(selected_indices)} '
            f'of {len(all_filenames)} segments. Example missing keys: {missing}'
        )

    return umap_coords[np.asarray(selected_indices, dtype=np.int64), :2]


def _resolve_day_file(base_dir, subject_name, day_str, suffix):
    """Resolve a per-day pickle path with flexible naming conventions.

    Supports both:
      - {subject}_{day_str}_subdir_{suffix}.pkl
      - {subject}_{day_str}_*_{suffix}.pkl (e.g., *_dph_{suffix}.pkl)
    """
    exact = os.path.join(base_dir, f'{subject_name}_{day_str}_subdir_{suffix}.pkl')
    if os.path.exists(exact):
        return exact

    pattern = os.path.join(base_dir, f'{subject_name}_{day_str}_*_{suffix}.pkl')
    matches = sorted(glob(pattern))
    if matches:
        return matches[0]

    return None


def _estimate_overlap_boundary_sec(spec, sr, hop_length,
                                   on1, dur1, on2, dur2, end1):
    """Estimate split boundary for partially overlapping segments.

    MATLAB-inspired behavior:
      - search starts halfway into first segment
      - search ends halfway into second segment
      - choose minimum-energy time bin in that window
    Fallback: midpoint of overlap window.
    """
    overlap_start = on2
    overlap_end = end1
    if overlap_end <= overlap_start:
        return 0.5 * (overlap_start + overlap_end)

    search_start = max(overlap_start, on1 + 0.5 * dur1)
    search_end = min(overlap_end, on2 + 0.5 * dur2)
    if search_end <= search_start:
        return 0.5 * (overlap_start + overlap_end)

    if spec is None or spec.size == 0:
        return 0.5 * (search_start + search_end)

    q = np.sum(spec.astype(np.float32), axis=0)
    if q.size == 0:
        return 0.5 * (search_start + search_end)

    c0 = int(np.floor((search_start * sr) / hop_length))
    c1 = int(np.ceil((search_end * sr) / hop_length))
    c0 = max(0, min(c0, q.size - 1))
    c1 = max(0, min(c1, q.size - 1))
    if c1 < c0:
        c0, c1 = c1, c0

    window = q[c0:c1 + 1]
    if window.size == 0:
        return 0.5 * (search_start + search_end)

    c_min = c0 + int(np.argmin(window))
    return float((c_min * hop_length) / sr)


def _fix_overlaps_iterative(file_ids, onset_sec, duration_sec, spectrograms,
                            sr, hop_length, max_passes=50):
    """Iteratively fix overlaps until none remain.

    Strategy (inspired by func_syl_check_overlap.m for WhisperSeg case):
      1) delete completely contained second segments
      2) split partial overlaps at low-energy boundary between neighboring segments
      3) repeat until no overlaps remain

    Returns:
      onset_fixed, duration_fixed, keep_mask, stats_dict
    """
    file_ids = np.asarray(file_ids, dtype=np.int32)
    on = np.asarray(onset_sec, dtype=np.float64).copy()
    dur = np.asarray(duration_sec, dtype=np.float64).copy()

    n = len(on)
    keep = np.ones(n, dtype=bool)
    eps = 1e-9

    stats = {
        'passes': 0,
        'deleted_contained': 0,
        'deleted_invalid': 0,
        'adjusted_partial': 0,
        'overlaps_initial': 0,
        'overlaps_final': 0,
    }

    def _count_overlaps(cur_keep, cur_on, cur_dur):
        count = 0
        for fid in np.unique(file_ids[cur_keep]):
            idx = np.flatnonzero(np.logical_and(cur_keep, file_ids == fid))
            if idx.size <= 1:
                continue
            order = idx[np.argsort(cur_on[idx], kind='mergesort')]
            ends = cur_on[order] + cur_dur[order]
            count += int(np.sum(cur_on[order][1:] < ends[:-1]))
        return count

    stats['overlaps_initial'] = _count_overlaps(keep, on, dur)

    for p in range(max_passes):
        changed = False

        for fid in np.unique(file_ids[keep]):
            idx = np.flatnonzero(np.logical_and(keep, file_ids == fid))
            if idx.size <= 1:
                continue

            order = idx[np.argsort(on[idx], kind='mergesort')]
            spec = spectrograms.get(int(fid))

            for t in range(len(order) - 1):
                j = int(order[t])
                k = int(order[t + 1])
                if not (keep[j] and keep[k]):
                    continue

                on1 = float(on[j])
                dur1 = float(dur[j])
                end1 = on1 + dur1

                on2 = float(on[k])
                dur2 = float(dur[k])
                end2 = on2 + dur2

                # Ignore already non-overlapping pairs
                if on2 >= end1 - eps:
                    continue

                # Delete fully contained second segment
                if end1 >= end2 - eps:
                    keep[k] = False
                    stats['deleted_contained'] += 1
                    changed = True
                    continue

                # Partial overlap: split at low-energy boundary
                boundary = _estimate_overlap_boundary_sec(
                    spec, sr, hop_length,
                    on1, dur1, on2, dur2, end1
                )
                boundary = min(max(boundary, on1), end2)

                new_end1 = boundary
                new_on2 = boundary
                new_dur1 = new_end1 - on1
                new_dur2 = end2 - new_on2

                if (new_dur1 <= eps) or (new_dur2 <= eps):
                    # If split collapses one segment, drop the shorter one
                    if dur1 <= dur2:
                        keep[j] = False
                    else:
                        keep[k] = False
                    stats['deleted_invalid'] += 1
                    changed = True
                    continue

                dur[j] = new_dur1
                on[k] = new_on2
                dur[k] = new_dur2
                stats['adjusted_partial'] += 1
                changed = True

        # Remove any invalid segments produced by previous edits
        invalid = np.logical_and(keep, np.logical_or(~np.isfinite(on), ~np.isfinite(dur)))
        invalid = np.logical_or(invalid, np.logical_and(keep, dur <= eps))
        n_invalid = int(np.sum(invalid))
        if n_invalid > 0:
            keep[invalid] = False
            stats['deleted_invalid'] += n_invalid
            changed = True

        stats['passes'] = p + 1
        if not changed:
            break

    stats['overlaps_final'] = _count_overlaps(keep, on, dur)
    return on.astype(np.float32), dur.astype(np.float32), keep, stats


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_to_hdf5(subject_name, data_dir, spectrograms_dir, segmentation_dir,
                   embedding_dir, umap_dir, output_path,
                   pitch_dir=None,
                   sr=32000, hop_length=128, n_pca_components=100,
                   embedding_style='first_timebin', n_neighbors=100, all_args=None):
    """Export all preprocessing results to HDF5 format for clustering_app.

    Args:
        subject_name: Subject identifier.
        data_dir: Directory containing wav files (directly or in day folders).
        spectrograms_dir: Directory with spectrogram pickle files.
        segmentation_dir: Directory with segmentation pickle files.
        embedding_dir: Directory with embedding pickle files.
        umap_dir: Directory with UMAP results.
        output_path: Path for output HDF5 file.
        pitch_dir: (optional) Directory with pitch pickle files.
        sr: Sample rate in Hz.
        hop_length: Hop length used for spectrograms.
        n_pca_components: Number of PCA components to compute from embeddings.
        embedding_style: Embedding style subfolder name.
        n_neighbors: UMAP n_neighbors (for loading the right file).
        all_args: Full argparse namespace (stored for reproducibility).
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    print('Collating data for HDF5 export...')

    all_days = get_all_days(data_dir)

    # ------------------------------------------------------------------
    # 1. Collate segment-level data from embedding pickles
    # ------------------------------------------------------------------
    all_file_ids = []
    all_filenames = []
    all_segment_ranges = []
    all_onset_sec = []
    all_duration_sec = []
    all_pitch_hz = []
    all_embeddings = []

    file_paths = []
    file_id_map = {}          # full_path → file_id
    current_file_id = 0

    spectrograms = {}         # file_id → int8 spectrogram
    pitch_data = {}           # file_id → (time, f0)
    n_invalid_segments_skipped = 0
    n_segments_outside_subset_skipped = 0

    for day in tqdm(all_days, desc='Loading day data'):
        day_str = get_day_string(day)

        # -- Load embeddings --
        emb_path = _resolve_day_file(
            os.path.join(embedding_dir, embedding_style),
            subject_name,
            day_str,
            'embeddings',
        )
        if emb_path is None:
            continue
        emb_dict = pickle.load(open(emb_path, 'rb'))

        # -- Load spectrograms --
        spec_path = _resolve_day_file(
            spectrograms_dir,
            subject_name,
            day_str,
            'spectrograms',
        )
        if spec_path is not None:
            spec_dict = pickle.load(open(spec_path, 'rb'))
            spec_by_fname = {os.path.basename(f): s
                             for f, s in zip(spec_dict['filename'],
                                             spec_dict['spectrogram'])}
        else:
            spec_by_fname = {}

        # -- Load pitch (optional) --
        if pitch_dir is not None:
            pitch_path = _resolve_day_file(
                pitch_dir,
                subject_name,
                day_str,
                'pitch',
            )
            if pitch_path is not None:
                pitch_dict = pickle.load(open(pitch_path, 'rb'))
                pitch_by_fname = {
                    os.path.basename(f): (t, p)
                    for f, t, p in zip(
                        pitch_dict.get('filename', []),
                        pitch_dict.get('timepoints', []),
                        pitch_dict.get('pitch', []),
                    )
                }
            else:
                pitch_by_fname = {}
        else:
            pitch_by_fname = {}

        # -- Wav base directory --
        if day is None:
            wav_base_dir = data_dir
        else:
            wav_base_dir = os.path.join(data_dir, day_str)

        allowed_wav_files = get_wav_files_for_day(data_dir, day)
        allowed_fnames = {os.path.basename(w) for w in allowed_wav_files}
        if len(allowed_fnames) == 0:
            continue

        # -- Iterate over segments --
        for fname, seg_range, embedding in zip(emb_dict['filename'],
                                                emb_dict['segment_range'],
                                                emb_dict['embedding']):
            fname_base = os.path.basename(str(fname))
            if fname_base not in allowed_fnames:
                n_segments_outside_subset_skipped += 1
                continue

            try:
                onset = float(seg_range[0])
                offset = float(seg_range[1])
            except Exception:
                n_invalid_segments_skipped += 1
                continue

            if not (np.isfinite(onset) and np.isfinite(offset)):
                n_invalid_segments_skipped += 1
                continue

            duration = offset - onset
            if duration <= 0:
                n_invalid_segments_skipped += 1
                continue

            # -- Assign file_id --
            full_path = os.path.join(wav_base_dir, fname_base)
            if full_path not in file_id_map:
                file_id_map[full_path] = current_file_id
                file_paths.append(full_path)

                # Store spectrogram for this file
                if fname_base in spec_by_fname:
                    spectrograms[current_file_id] = spec_by_fname[fname_base]
                else:
                    # Fallback: compute spectrogram from wav
                    try:
                        y, _ = librosa.load(full_path, sr=sr)
                        n_fft = int(getattr(all_args, 'n_fft', 512)) if all_args else 512
                        hop = int(getattr(all_args, 'hop_length', hop_length)) if all_args else hop_length
                        f_min = int(getattr(all_args, 'spec_min_freq', 312)) if all_args else 312
                        f_max = int(getattr(all_args, 'spec_max_freq', 8000)) if all_args else 8000

                        D = librosa.amplitude_to_db(
                            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)),
                            ref=np.max)
                        max_abs = np.max(np.abs(D))
                        if max_abs > 0:
                            D = (D / max_abs * 127).astype(np.int8)
                        else:
                            D = np.zeros_like(D, dtype=np.int8)

                        f = np.arange(D.shape[0]) * sr / n_fft
                        f_window = np.logical_and(f_min < f, f < f_max)
                        D = D[f_window, :]
                        spectrograms[current_file_id] = D
                    except Exception:
                        pass

                # Store pitch for this file
                if fname_base in pitch_by_fname:
                    pitch_data[current_file_id] = pitch_by_fname[fname_base]

                current_file_id += 1

            file_id = file_id_map[full_path]

            # Compute per-segment median pitch
            median_pitch = np.nan
            pitch_tuple = pitch_data.get(file_id)
            if pitch_tuple is not None:
                timepoints, f0_vals = pitch_tuple
                timepoints = np.asarray(timepoints)
                f0_vals = np.asarray(f0_vals)
                in_seg = np.logical_and(timepoints >= onset, timepoints <= offset)
                if np.any(in_seg):
                    pitch_seg = f0_vals[in_seg]
                    valid_pitch = np.logical_and(np.isfinite(pitch_seg), pitch_seg > 0)
                    if np.any(valid_pitch):
                        median_pitch = float(np.median(pitch_seg[valid_pitch]))

            all_file_ids.append(file_id)
            all_filenames.append(fname_base)
            all_segment_ranges.append((onset, offset))
            all_onset_sec.append(onset)
            all_duration_sec.append(duration)
            all_pitch_hz.append(median_pitch)
            all_embeddings.append(embedding)

    if len(all_embeddings) == 0:
        print('No data found to export!')
        return

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f'Total segments: {len(all_embeddings)}')
    if n_invalid_segments_skipped > 0:
        print(f'Warning: Skipped {n_invalid_segments_skipped} invalid segments '
              '(non-finite or non-positive duration).')
    if n_segments_outside_subset_skipped > 0:
        print(f'Filtered out {n_segments_outside_subset_skipped} segments not present '
              f'in data_dir subset ({data_dir}).')
    print(f'Embedding shape: {embeddings_array.shape}')

    # ------------------------------------------------------------------
    # 2. Load and validate UMAP coordinates
    # ------------------------------------------------------------------
    umap_path = os.path.join(umap_dir,
                             f'umap_coordinates_{n_neighbors}neighbors.pickle')
    if not os.path.exists(umap_path):
        raise FileNotFoundError(
            f'UMAP coordinates not found at {umap_path}. '
            'Run --compute-umap first.'
        )

    umap_dict = pickle.load(open(umap_path, 'rb'))
    umap_total = np.asarray(umap_dict.get('umap_coordinates', []), dtype=np.float32)
    if umap_total.shape[0] == len(all_embeddings):
        umap_coords, umap_warnings = _validate_umap_alignment(
            all_filenames, all_onset_sec, umap_dict, len(all_embeddings)
        )
        for w in umap_warnings:
            print(f'  [UMAP] {w}')
    else:
        print('UMAP appears to come from a larger embedding pool; subsetting by '
              '(filename, onset, offset)...')
        umap_coords = _subset_umap_by_segment_ranges(
            all_filenames, all_segment_ranges, umap_dict
        )
    print(f'Loaded UMAP coordinates: {umap_coords.shape}')

    # ------------------------------------------------------------------
    # 3. Iterative overlap fixing (WhisperSeg-inspired)
    # ------------------------------------------------------------------
    print('Checking/fixing overlaps (iterative)...')
    onset_fixed, duration_fixed, keep_mask, ov_stats = _fix_overlaps_iterative(
        np.asarray(all_file_ids, dtype=np.int32),
        np.asarray(all_onset_sec, dtype=np.float32),
        np.asarray(all_duration_sec, dtype=np.float32),
        spectrograms,
        sr=sr,
        hop_length=hop_length,
    )

    kept = int(np.sum(keep_mask))
    removed = int(len(keep_mask) - kept)
    print(
        f"  overlaps: initial={ov_stats['overlaps_initial']} final={ov_stats['overlaps_final']} "
        f"passes={ov_stats['passes']} adjusted={ov_stats['adjusted_partial']} "
        f"removed={removed}"
    )

    if removed > 0 or ov_stats['adjusted_partial'] > 0:
        keep_idx = np.flatnonzero(keep_mask)
        all_file_ids = [all_file_ids[i] for i in keep_idx]
        all_filenames = [all_filenames[i] for i in keep_idx]
        all_pitch_hz = [all_pitch_hz[i] for i in keep_idx]
        all_onset_sec = onset_fixed[keep_idx].tolist()
        all_duration_sec = duration_fixed[keep_idx].tolist()

        embeddings_array = embeddings_array[keep_idx, :]
        umap_coords = umap_coords[keep_idx, :]

    # ------------------------------------------------------------------
    # 4. PCA on embeddings (after overlap fixing)
    # ------------------------------------------------------------------
    print(f'Computing PCA ({n_pca_components} components)...')
    pca = PCA(n_components=min(n_pca_components, embeddings_array.shape[1]))
    pca_coords = pca.fit_transform(embeddings_array)
    print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')

    # ------------------------------------------------------------------
    # 5. Write HDF5
    # ------------------------------------------------------------------
    print(f'Writing HDF5 to {output_path}...')
    with h5py.File(output_path, 'w') as f:
        # -- segments --
        seg_grp = f.create_group('segments')
        seg_grp.create_dataset('segment_id', data=np.arange(len(all_file_ids)))
        seg_grp.create_dataset('file_id',
                               data=np.array(all_file_ids, dtype=np.int32))
        seg_grp.create_dataset('onset_sec',
                               data=np.array(all_onset_sec, dtype=np.float32))
        seg_grp.create_dataset('duration_sec',
                               data=np.array(all_duration_sec, dtype=np.float32))
        seg_grp.create_dataset('pitch_hz',
                               data=np.array(all_pitch_hz, dtype=np.float32))
        seg_grp.create_dataset('umap',
                               data=umap_coords.astype(np.float32))
        seg_grp.create_dataset('cluster_id',
                               data=np.zeros(len(all_file_ids), dtype=np.int32))
        seg_grp.create_dataset('pca',
                               data=pca_coords.astype(np.float32))

        # -- files --
        files_grp = f.create_group('files')
        files_grp.create_dataset('file_id',
                                 data=np.arange(len(file_paths), dtype=np.int32))
        filenames = [os.path.basename(p) for p in file_paths]
        filename_bytes = np.array([fn.encode('utf-8') for fn in filenames],
                                  dtype='S')
        files_grp.create_dataset('filename', data=filename_bytes)

        # -- embeddings (raw) --
        emb_grp = f.create_group('embeddings')
        emb_grp.create_dataset('segment_id',
                               data=np.arange(len(all_file_ids), dtype=np.int32))
        emb_grp.create_dataset('raw',
                               data=embeddings_array, compression='gzip')

        # -- spectrograms --
        spec_grp = f.create_group('spectrograms')
        spec_grp.create_dataset('file_id',
                                data=np.array(list(spectrograms.keys()),
                                              dtype=np.int32))
        for file_id, spec in tqdm(spectrograms.items(),
                                  desc='Writing spectrograms'):
            spec_grp.create_dataset(str(file_id), data=spec,
                                    compression='gzip')

        # -- pitch --
        pitch_grp = f.create_group('pitch')
        pitch_grp.create_dataset('file_id',
                                 data=np.array(list(pitch_data.keys()),
                                               dtype=np.int32))
        for file_id, (timepoints, f0) in pitch_data.items():
            file_grp = pitch_grp.create_group(str(file_id))
            file_grp.create_dataset('time',
                                    data=np.asarray(timepoints, dtype=np.float32),
                                    compression='gzip')
            file_grp.create_dataset('f0',
                                    data=np.asarray(f0, dtype=np.float32),
                                    compression='gzip')

        # -- parameters --
        params_grp = f.create_group('parameters')

        if all_args is not None:
            args_dict = vars(all_args)
            param_map = {
                'subject': 'subject',
                'data_dir': 'data_dir',
                'output_dir': 'output_dir',
                'device': 'device',
                'sr': 'audio_sr',
                'n_fft': 'spec_n_fft',
                'hop_length': 'spec_hop_length',
                'spec_min_freq': 'spec_min_freq',
                'spec_max_freq': 'spec_max_freq',
                'pitch_floor': 'pitch_floor',
                'segmenter_model': 'seg_model',
                'seg_min_freq': 'seg_min_freq',
                'seg_time_step': 'seg_time_step',
                'seg_min_length': 'seg_min_length',
                'seg_eps': 'seg_eps',
                'seg_num_trials': 'seg_num_trials',
                'embedder_model': 'emb_model',
                'emb_min_freq': 'emb_min_freq',
                'emb_time_step': 'emb_time_step',
                'emb_num_trials': 'emb_num_trials',
                'embedding_style': 'emb_style',
                'batch_size': 'emb_batch_size',
                'n_neighbors': 'umap_n_neighbors',
                'train_percentage': 'umap_train_percentage(0-1)',
                'n_pca': 'pca_n_components',
            }
            for arg_name, param_name in param_map.items():
                if arg_name in args_dict:
                    val = args_dict[arg_name]
                    if val is not None and isinstance(val, (str, int, float, bool)):
                        params_grp.attrs[param_name] = val

        # Derived parameters (always present)
        params_grp.attrs['sr_processing'] = sr
        params_grp.attrs['hop_length'] = hop_length
        params_grp.attrs['n_pca_components'] = n_pca_components
        params_grp.attrs['n_neighbors'] = n_neighbors
        params_grp.attrs['subject'] = subject_name

        # Detect original wav sample rate
        if file_paths:
            import wave
            try:
                with wave.open(file_paths[0], 'rb') as wf:
                    params_grp.attrs['sr_original'] = wf.getframerate()
            except Exception:
                params_grp.attrs['sr_original'] = sr

        # UMAP map range for visualization
        umap_min = umap_coords.min(axis=0)
        umap_max = umap_coords.max(axis=0)
        umap_range = np.array([umap_min[0], umap_max[0],
                               umap_min[1], umap_max[1]])
        f.create_dataset('umap_maprange', data=umap_range)

    print(f'HDF5 export complete: {output_path}')
    print(f'  - {len(all_file_ids)} segments')
    print(f'  - {len(file_paths)} files')
    print(f'  - {len(spectrograms)} spectrograms')
    print(f'  - {len(pitch_data)} pitch tracks')

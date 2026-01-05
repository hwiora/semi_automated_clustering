#!/usr/bin/env python
"""
Audio Preprocessing Pipeline

A pipeline for preprocessing audio recordings with vocalization segmentation,
embedding extraction, UMAP clustering, and HDF5 export for semi-automated clustering.

Usage:
    python preprocess.py --subject BIRD_ID --data-dir ./data --output-dir ./output --all
    python preprocess.py --subject BIRD_ID --data-dir ./data --output-dir ./output \
        --compute-spectrograms --compute-segmentations --compute-embeddings \
        --compute-umap --export-hdf5
"""
import argparse
import os
import sys
import warnings

# Suppress warnings from ctranslate2/pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module='ctranslate2')
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def main():
    parser = argparse.ArgumentParser(
        description='Audio preprocessing pipeline for vocalization analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps and export to HDF5:
  python preprocess.py --subject R2764 --data-dir ./data/R2764 --output-dir ./output/R2764 --all

  # Run individual steps:
  python preprocess.py --subject R2764 --data-dir ./data/R2764 --output-dir ./output/R2764 \\
      --compute-spectrograms --compute-segmentations --compute-embeddings \\
      --compute-umap --export-hdf5

  # Just export existing processed data to HDF5:
  python preprocess.py --subject R2764 --data-dir ./data/R2764 --output-dir ./output/R2764 \\
      --export-hdf5
        """
    )
    
    # Required arguments
    parser.add_argument('--subject', required=True,
                        help='Subject identifier (used in output filenames)')
    parser.add_argument('--data-dir', required=True,
                        help='Path to input audio data directory (contains day folders or generic subdirectories with .wav files)')
    parser.add_argument('--output-dir', required=True,
                        help='Path for output files')
    
    # Convenience flag
    parser.add_argument('--all', action='store_true',
                        help='Run all processing steps including HDF5 export')
    
    # Step selection flags
    parser.add_argument('--compute-spectrograms', action='store_true',
                        help='Compute spectrograms from audio files')
    parser.add_argument('--compute-pitch', action='store_true',
                        help='Extract pitch contours from audio files')
    parser.add_argument('--compute-segmentations', action='store_true',
                        help='Run WhisperSeg segmentation')
    parser.add_argument('--compute-embeddings', action='store_true',
                        help='Compute embeddings from segmented audio')
    parser.add_argument('--compute-umap', action='store_true',
                        help='Compute UMAP projection from embeddings')
    parser.add_argument('--export-hdf5', action='store_true',
                        help='Export all results to HDF5 for clustering_app')
    
    # Audio parameters
    parser.add_argument('--sr', type=int, default=32000,
                        help='Sample rate in Hz (default: 32000)')
    
    # Spectrogram parameters
    parser.add_argument('--n-fft', type=int, default=512,
                        help='FFT size in samples (default: 512)')
    parser.add_argument('--hop-length', type=int, default=128,
                        help='Hop length in samples (default: 128)')
    parser.add_argument('--spec-min-freq', type=int, default=312,
                        help='Minimum frequency for spectrograms in Hz (default: 312)')
    parser.add_argument('--spec-max-freq', type=int, default=8000,
                        help='Maximum frequency for spectrograms in Hz (default: 8000)')
    
    # Pitch parameters
    parser.add_argument('--pitch-floor', type=int, default=312,
                        help='Pitch floor in Hz (default: 312)')
    
    # WhisperSeg parameters
    parser.add_argument('--segmenter-model', default='nccratliri/whisperseg-base-animal-vad-ct2',
                        help='WhisperSeg model path (default: nccratliri/whisperseg-base-animal-vad-ct2)')
    parser.add_argument('--seg-min-freq', type=int, default=312,
                        help='Minimum frequency for segmentation (default: 312)')
    parser.add_argument('--seg-time-step', type=float, default=0.002,
                        help='Spectrogram time step for segmentation in seconds (default: 0.002)')
    parser.add_argument('--seg-min-length', type=float, default=0.01,
                        help='Minimum segment length in seconds (default: 0.01)')
    parser.add_argument('--seg-eps', type=float, default=0.02,
                        help='DBSCAN epsilon for segment consolidation (default: 0.02)')
    parser.add_argument('--seg-num-trials', type=int, default=3,
                        help='Number of segmentation trials (default: 3)')
    
    # Embedding parameters
    parser.add_argument('--embedder-model', default='Systran/faster-whisper-large-v3',
                        help='Embedding model path (default: Systran/faster-whisper-large-v3)')
    parser.add_argument('--emb-min-freq', type=int, default=0,
                        help='Minimum frequency for embeddings (default: 0)')
    parser.add_argument('--emb-time-step', type=float, default=0.0025,
                        help='Spectrogram time step for embeddings in seconds (default: 0.0025)')
    parser.add_argument('--emb-num-trials', type=int, default=3,
                        help='Number of embedding trials (default: 3)')
    parser.add_argument('--embedding-style', default='first_timebin',
                        help='Embedding extraction style (default: first_timebin)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding computation (default: 32)')
    
    # UMAP parameters
    parser.add_argument('--n-neighbors', type=int, default=100,
                        help='Number of neighbors for UMAP (default: 100)')
    parser.add_argument('--train-percentage', type=float, default=1.0,
                        help='Fraction of data to use for UMAP fitting (default: 1.0)')
    
    # HDF5 export parameters
    parser.add_argument('--n-pca', type=int, default=100,
                        help='Number of PCA components to include (default: 100)')
    parser.add_argument('--hdf5-output', type=str, default=None,
                        help='Custom HDF5 output path (default: {output_dir}/{subject}.h5)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for ML models: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.compute_spectrograms = True
        args.compute_pitch = True
        args.compute_segmentations = True
        args.compute_embeddings = True
        args.compute_umap = True
        args.export_hdf5 = True
    
    # Validate that at least one step is selected
    steps_selected = any([
        args.compute_spectrograms,
        args.compute_pitch,
        args.compute_segmentations,
        args.compute_embeddings,
        args.compute_umap,
        args.export_hdf5,
    ])
    
    if not steps_selected:
        parser.error('No processing steps selected. Use --all or select individual steps. Use --help for options.')
    
    # Define output directories
    spectrograms_dir = os.path.join(args.output_dir, 'Spectrograms')
    pitch_dir = os.path.join(args.output_dir, 'PitchComputation')
    segmentation_dir = os.path.join(args.output_dir, 'WhisperSeg')
    embedding_dir = os.path.join(args.output_dir, 'Embeddings')
    umap_dir = os.path.join(args.output_dir, 'UMAP')
    
    # Default HDF5 output path
    if args.hdf5_output is None:
        args.hdf5_output = os.path.join(args.output_dir, f'{args.subject}.h5')
    
    print('=' * 60)
    print('AUDIO PREPROCESSING PIPELINE')
    print('=' * 60)
    print(f'Subject: {args.subject}')
    print(f'Data directory: {args.data_dir}')
    print(f'Output directory: {args.output_dir}')
    print()
    
    step_num = 1
    
    # Run selected steps
    if args.compute_spectrograms:
        from steps.compute_spectrogram import compute_spectrograms
        print(f'[STEP {step_num}] Computing spectrograms...')
        print('-' * 60)
        compute_spectrograms(
            args.subject,
            args.data_dir,
            spectrograms_dir,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            min_freq=args.spec_min_freq,
            max_freq=args.spec_max_freq,
        )
        step_num += 1
        print()
    
    if args.compute_pitch:
        from steps.compute_pitch import compute_pitch
        print(f'[STEP {step_num}] Computing pitch...')
        print('-' * 60)
        compute_pitch(
            args.subject,
            args.data_dir,
            pitch_dir,
            sr=args.sr,
            pitch_floor=args.pitch_floor,
        )
        step_num += 1
        print()
    
    if args.compute_segmentations:
        from steps.run_whisperseg import run_whisperseg
        print(f'[STEP {step_num}] Running WhisperSeg segmentation...')
        print('-' * 60)
        run_whisperseg(
            args.subject,
            args.data_dir,
            segmentation_dir,
            model_path=args.segmenter_model,
            sr=args.sr,
            min_freq=args.seg_min_freq,
            spec_time_step=args.seg_time_step,
            min_segment_length=args.seg_min_length,
            eps=args.seg_eps,
            num_trials=args.seg_num_trials,
            device=args.device,
        )
        step_num += 1
        print()
    
    if args.compute_embeddings:
        from steps.compute_embeddings import compute_embeddings
        print(f'[STEP {step_num}] Computing embeddings...')
        print('-' * 60)
        compute_embeddings(
            args.subject,
            args.data_dir,
            segmentation_dir,
            embedding_dir,
            model_path=args.embedder_model,
            sr=args.sr,
            min_freq=args.emb_min_freq,
            spec_time_step=args.emb_time_step,
            num_trials=args.emb_num_trials,
            embedding_style=args.embedding_style,
            batch_size=args.batch_size,
            device=args.device,
        )
        step_num += 1
        print()
    
    if args.compute_umap:
        from steps.compute_umap import compute_umap
        print(f'[STEP {step_num}] Computing UMAP...')
        print('-' * 60)
        embedding_style_dir = os.path.join(embedding_dir, args.embedding_style)
        compute_umap(
            embedding_style_dir,
            umap_dir,
            n_neighbors=args.n_neighbors,
            train_percentage=args.train_percentage,
        )
        step_num += 1
        print()
    
    if args.export_hdf5:
        from steps.export_hdf5 import export_to_hdf5
        print(f'[STEP {step_num}] Exporting to HDF5...')
        print('-' * 60)
        export_to_hdf5(
            args.subject,
            args.data_dir,
            spectrograms_dir,
            segmentation_dir,
            embedding_dir,
            umap_dir,
            args.hdf5_output,
            sr=args.sr,
            hop_length=args.hop_length,
            n_pca_components=args.n_pca,
            embedding_style=args.embedding_style,
            n_neighbors=args.n_neighbors,
            all_args=args,  # Pass all arguments for reproducibility
        )
        step_num += 1
        print()
    
    print('=' * 60)
    print('PIPELINE COMPLETE!')
    print('=' * 60)
    if args.export_hdf5:
        print(f'HDF5 output: {args.hdf5_output}')
        print(f'Run clustering: python ../clustering_app.py {args.hdf5_output}')


if __name__ == '__main__':
    main()

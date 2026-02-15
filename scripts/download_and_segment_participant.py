# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Download a single participant (file_id) from the Seamless Interaction dataset,
then segment video/audio by contiguous uniform dominant emotion (3–10 s segments)
for downstream model evaluation/finetuning (e.g. Qwen3-Omni, NVIDIA Omnivinci).

Usage (from repo root, with package installed):
  pip install -e .
  python scripts/download_and_segment_participant.py --file_id V00_S0809_I00000309_P0947

Requires: ffmpeg on PATH for video/audio segment extraction.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo src to path when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS

# Emotion categories from movement:emotion_scores (index 0–7)
EMOTION_NAMES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

MIN_SEGMENT_FRAMES = 90   # 3 s at 30 Hz
MAX_SEGMENT_FRAMES = 300  # 10 s at 30 Hz
DEFAULT_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a participant and segment by uniform dominant emotion (3–10 s)."
    )
    parser.add_argument(
        "--file_id",
        type=str,
        required=True,
        help="File ID (e.g. V00_S0809_I00000309_P0947) from assets/filelist.csv",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=str(_REPO_ROOT / "assets" / "filelist.csv"),
        help="Path to filelist.csv",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=str(Path.home() / "datasets" / "seamless_interaction"),
        help="Local dataset root (download and segments go here)",
    )
    parser.add_argument(
        "--require_imitator_movement",
        action="store_true",
        default=True,
        help="Require has_imitator_movement=1 for file_id (default: True)",
    )
    parser.add_argument(
        "--no_require_imitator_movement",
        action="store_false",
        dest="require_imitator_movement",
        help="Do not require has_imitator_movement",
    )
    return parser.parse_args()


def validate_file_id_in_filelist(file_id: str, filelist_path: str, require_imitator: bool) -> pd.Series:
    """Load filelist and return the row for file_id; raise if missing or invalid."""
    if not os.path.isfile(filelist_path):
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")
    df = pd.read_csv(filelist_path)
    row = df[df["file_id"] == file_id]
    if row.empty:
        raise ValueError(f"file_id {file_id} not found in {filelist_path}")
    r = row.iloc[0]
    if require_imitator and r.get("has_imitator_movement", 0) != 1:
        raise ValueError(
            f"file_id {file_id} has has_imitator_movement={r.get('has_imitator_movement')}; "
            "emotion features require has_imitator_movement=1. Use --no_require_imitator_movement to skip."
        )
    return r


def download_participant(
    file_id: str,
    filelist_path: str,
    local_dir: str,
    require_imitator_movement: bool,
) -> str:
    """Download participant from S3 and return base directory path. Raises on failure."""
    row = validate_file_id_in_filelist(file_id, filelist_path, require_imitator_movement)
    label = str(row["label"])
    split = str(row["split"])
    batch_idx = int(row["batch_idx"])
    archive_idx = int(row["archive_idx"])

    config = DatasetConfig(
        label=label,
        split=split,
        local_dir=local_dir,
        preferred_vendors_only=True,
    )
    fs = SeamlessInteractionFS(config=config, filelist_path=filelist_path)
    print(f"Downloading {file_id} from S3...")
    fs.gather_file_id_data_from_s3(file_id)

    base_path = os.path.join(
        local_dir, label, split, f"{batch_idx:04d}", f"{archive_idx:04d}"
    )
    mp4_path = os.path.join(base_path, f"{file_id}.mp4")
    wav_path = os.path.join(base_path, f"{file_id}.wav")
    npz_path = os.path.join(base_path, f"{file_id}.npz")

    missing = []
    if not os.path.isfile(mp4_path):
        missing.append(f"{file_id}.mp4")
    if not os.path.isfile(wav_path):
        missing.append(f"{file_id}.wav")
    if not os.path.isfile(npz_path):
        missing.append(f"{file_id}.npz")
    if missing:
        raise RuntimeError(f"Download incomplete: missing {missing} in {base_path}")

    print(f"Download complete. Files in {base_path}")
    return base_path


def load_emotion_arrays(npz_path: str):
    """Load emotion_scores (and optionally valence/arousal) from NPZ. Return (scores, T)."""
    data = np.load(npz_path)
    key_scores = "movement:emotion_scores"
    if key_scores not in data:
        raise KeyError(
            f"NPZ at {npz_path} missing required key '{key_scores}'. "
            "Ensure the file has imitator movement features."
        )
    scores = np.asarray(data[key_scores])
    if scores.ndim == 1:
        scores = np.expand_dims(scores, -1)
    # (T, 8) for emotion_scores
    if scores.ndim != 2 or scores.shape[-1] != 8:
        raise ValueError(
            f"Expected movement:emotion_scores shape (T, 8), got {scores.shape}"
        )
    T = scores.shape[0]
    return data, scores, T


def compute_dominant_emotion_per_frame(scores: np.ndarray) -> np.ndarray:
    """(T, 8) -> (T,) int array of dominant emotion index per frame."""
    return np.argmax(scores, axis=-1).astype(np.int32)


def find_uniform_emotion_runs(dominant: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Find contiguous runs of same dominant emotion.
    Returns list of (start_frame, end_frame, emotion_index).
    """
    runs = []
    i = 0
    T = len(dominant)
    while i < T:
        em = int(dominant[i])
        j = i + 1
        while j < T and int(dominant[j]) == em:
            j += 1
        runs.append((i, j, em))
        i = j
    return runs


def split_runs_into_segments(
    runs: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """
    Apply length rules: drop < 3 s, keep 3–10 s as one segment, break > 10 s into 10 s chunks.
    Returns list of (start_frame, end_frame, emotion_index).
    """
    segments = []
    for start, end, em in runs:
        n = end - start
        if n < MIN_SEGMENT_FRAMES:
            continue
        if n <= MAX_SEGMENT_FRAMES:
            segments.append((start, end, em))
            continue
        # Break into 10 s (300-frame) chunks; remainder >= 3 s kept, else dropped
        s = start
        while s < end:
            seg_end = min(s + MAX_SEGMENT_FRAMES, end)
            if seg_end - s >= MIN_SEGMENT_FRAMES:
                segments.append((s, seg_end, em))
            s = seg_end
    return segments


def get_video_fps_and_frame_count(video_path: str) -> tuple[float, int]:
    """Return (fps, frame_count) using OpenCV."""
    try:
        import cv2
    except ImportError:
        return DEFAULT_FPS, -1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return DEFAULT_FPS, -1
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), count


def extract_video_segment_ffmpeg(
    input_mp4: str,
    output_mp4: str,
    start_time: float,
    end_time: float,
) -> None:
    """Extract [start_time, end_time] from input_mp4 into output_mp4 using ffmpeg."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        input_mp4,
        "-t",
        str(duration),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        output_mp4,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio_segment_ffmpeg(
    input_wav: str,
    output_wav: str,
    start_time: float,
    end_time: float,
) -> None:
    """Extract [start_time, end_time] from input_wav into output_wav using ffmpeg."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        input_wav,
        "-t",
        str(duration),
        "-acodec",
        "copy",
        output_wav,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def slice_npz_by_frames(
    npz_path: str,
    start_frame: int,
    end_frame: int,
    segment_dir: str,
) -> None:
    """
    Load NPZ, slice every per-frame array (first dim >= end_frame) to [start_frame:end_frame],
    save as .npy (key with ':' replaced by '_').
    """
    data = np.load(npz_path)
    os.makedirs(segment_dir, exist_ok=True)
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim >= 1 and arr.shape[0] >= end_frame:
            sl = arr[start_frame:end_frame]
            base_name = key.replace(":", "_")
            npy_path = os.path.join(segment_dir, f"{base_name}.npy")
            np.save(npy_path, sl)


def write_segment_metadata(
    segment_dir: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    dominant_emotion_index: int,
) -> None:
    metadata = {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": start_time,
        "end_time": end_time,
        "dominant_emotion_index": int(dominant_emotion_index),
        "dominant_emotion_name": EMOTION_NAMES[int(dominant_emotion_index)],
    }
    path = os.path.join(segment_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_segments(
    base_path: str,
    file_id: str,
    segments: list[tuple[int, int, int]],
    fps: float,
) -> str:
    """
    For each segment: extract video/audio, slice NPZ, write metadata.
    Returns path to segments directory.
    """
    mp4_path = os.path.join(base_path, f"{file_id}.mp4")
    wav_path = os.path.join(base_path, f"{file_id}.wav")
    npz_path = os.path.join(base_path, f"{file_id}.npz")
    segments_dir = os.path.join(base_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    for idx, (start_frame, end_frame, emotion_index) in enumerate(segments):
        start_time = start_frame / fps
        end_time = end_frame / fps
        seg_name = f"segment_{idx:03d}"
        segment_dir = os.path.join(segments_dir, seg_name)
        os.makedirs(segment_dir, exist_ok=True)

        out_mp4 = os.path.join(segment_dir, "video.mp4")
        out_wav = os.path.join(segment_dir, "audio.wav")
        extract_video_segment_ffmpeg(mp4_path, out_mp4, start_time, end_time)
        extract_audio_segment_ffmpeg(wav_path, out_wav, start_time, end_time)
        slice_npz_by_frames(npz_path, start_frame, end_frame, segment_dir)
        write_segment_metadata(
            segment_dir, start_frame, end_frame, start_time, end_time, emotion_index
        )
        print(f"  Wrote {seg_name} ({start_time:.2f}s–{end_time:.2f}s, {EMOTION_NAMES[emotion_index]})")

    return segments_dir


def delete_large_files(base_path: str, file_id: str) -> None:
    """Remove full-length .mp4, .wav, .npz. Keep .json."""
    for ext in (".mp4", ".wav", ".npz"):
        path = os.path.join(base_path, f"{file_id}{ext}")
        if os.path.isfile(path):
            os.remove(path)
            print(f"Removed {path}")


def main() -> None:
    args = parse_args()
    file_id = args.file_id
    filelist_path = args.filelist
    local_dir = args.local_dir
    require_imitator = args.require_imitator_movement

    base_path = download_participant(
        file_id, filelist_path, local_dir, require_imitator
    )
    npz_path = os.path.join(base_path, f"{file_id}.npz")
    mp4_path = os.path.join(base_path, f"{file_id}.mp4")

    data, scores, T = load_emotion_arrays(npz_path)
    fps, video_frames = get_video_fps_and_frame_count(mp4_path)
    if video_frames > 0 and video_frames != T:
        T_use = min(T, video_frames)
        print(f"Note: NPZ frames={T}, video frames={video_frames}; using first {T_use} frames for alignment.")
        T = T_use
        scores = scores[:T_use]

    dominant = compute_dominant_emotion_per_frame(scores)
    runs = find_uniform_emotion_runs(dominant)
    segments = split_runs_into_segments(runs)

    if not segments:
        print("No segments of 3–10 s with uniform dominant emotion found. Exiting.")
        return

    print(f"Found {len(segments)} segments (3–10 s, uniform dominant emotion).")
    segments_dir = process_segments(base_path, file_id, segments, fps)
    delete_large_files(base_path, file_id)
    print("All done. Segments saved under {}; original video/audio/npz removed.".format(segments_dir))


if __name__ == "__main__":
    main()

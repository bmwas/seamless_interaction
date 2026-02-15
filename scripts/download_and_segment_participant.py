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

Interviewee-only: Dataset assets do not provide which participant is interviewer vs
interviewee (no A/B mapping). Use --interviewee_file_ids or --interviewee_list only
when you have an external mapping (e.g. from dataset maintainers or your own labeling).
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
# Minimum fraction of segment duration that must be speech (person talking) to keep segment.
MIN_SPEECH_FRACTION = 0.5
# Minimum length of continuous speech (no pause) required in a segment (seconds).
MIN_CONTINUOUS_SPEECH_SEC = 3.0
# Max gap between speech intervals to treat as same turn (seconds). Gaps below this = thinking pause; above = waiting for interviewer.
VAD_MERGE_GAP_SEC = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a participant and segment by uniform dominant emotion (3–10 s)."
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default=None,
        help="File ID (e.g. V00_S0809_I00000309_P0947). If omitted, use first subject from filelist (reference for later looping over all).",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=str(_REPO_ROOT / "assets" / "filelist.csv"),
        help="Path to filelist.csv (default: assets/filelist.csv)",
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
    parser.add_argument(
        "--interviewee_file_ids",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Path to CSV with file_id column (and optionally role). Only these file_ids are considered (e.g. interviewee-only). Requires external mapping; assets do not provide A/B.",
    )
    parser.add_argument(
        "--interviewee_list",
        type=str,
        default=None,
        metavar="TXT_PATH",
        help="Path to plain text file with one file_id per line. Only these file_ids are considered (e.g. interviewee-only). Requires external mapping.",
    )
    parser.add_argument(
        "--min_speech_fraction",
        type=float,
        default=0.5,
        metavar="F",
        help="Minimum fraction of segment duration that must be speech (person talking). 0.5 = must be talking 50%% of the time (default). Use a lower value (e.g. 0.3) if no segments remain.",
    )
    parser.add_argument(
        "--min_continuous_speech",
        type=float,
        default=MIN_CONTINUOUS_SPEECH_SEC,
        metavar="SEC",
        help=f"Minimum length of continuous speech in segment, in seconds (default: {MIN_CONTINUOUS_SPEECH_SEC:.0f}). Segments without this much speech (after merging thinking pauses) are dropped.",
    )
    parser.add_argument(
        "--vad_merge_gap",
        type=float,
        default=VAD_MERGE_GAP_SEC,
        metavar="SEC",
        help=f"Max gap between speech intervals to merge as same turn, in seconds (default: {VAD_MERGE_GAP_SEC:.1f}). Gaps below this = thinking pause (merged); above = waiting for other person (split).",
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


def load_allowed_file_ids(
    interviewee_file_ids_path: str | None,
    interviewee_list_path: str | None,
) -> set[str] | None:
    """
    Load set of allowed file_ids from CSV (file_id column) or plain list (one per line).
    Returns None if neither path is set. Raises if both are set (use one).
    """
    if interviewee_file_ids_path is not None and interviewee_list_path is not None:
        raise ValueError(
            "Use only one of --interviewee_file_ids or --interviewee_list, not both."
        )
    if interviewee_file_ids_path is None and interviewee_list_path is None:
        return None
    path = interviewee_file_ids_path or interviewee_list_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Interviewee file list not found: {path}")
    allowed = set()
    if interviewee_file_ids_path is not None:
        df = pd.read_csv(path)
        if "file_id" not in df.columns:
            raise ValueError(f"CSV {path} must have a 'file_id' column.")
        allowed = set(df["file_id"].astype(str).str.strip())
    else:
        with open(path) as f:
            for line in f:
                fid = line.strip()
                if fid:
                    allowed.add(fid)
    return allowed


def get_first_file_id(
    filelist_path: str,
    require_imitator: bool,
    allowed_file_ids: set[str] | None = None,
) -> str:
    """Return the first file_id from filelist (optionally with has_imitator_movement==1 and in allowed_file_ids)."""
    if not os.path.isfile(filelist_path):
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")
    df = pd.read_csv(filelist_path)
    if require_imitator:
        df = df[df.get("has_imitator_movement", 0) == 1]
    if allowed_file_ids is not None:
        df = df[df["file_id"].astype(str).isin(allowed_file_ids)]
    if df.empty:
        raise ValueError(
            "No rows in filelist matching criteria (has_imitator_movement and/or interviewee list). "
            "Use --no_require_imitator_movement or check --interviewee_file_ids/--interviewee_list."
        )
    return str(df.iloc[0]["file_id"])


def _ensure_directory_writable(base_path: str) -> None:
    """Ensure base_path exists and is writable; raise with a clear message if not."""
    os.makedirs(base_path, exist_ok=True)
    probe = os.path.join(base_path, ".write_probe")
    try:
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
    except OSError as e:
        raise PermissionError(
            f"Cannot write to {base_path}: {e}. "
            "Fix permissions (e.g. sudo chown -R $USER ...) and retry."
        ) from e


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

    base_path = os.path.join(
        local_dir, label, split, f"{batch_idx:04d}", f"{archive_idx:04d}"
    )
    _ensure_directory_writable(base_path)

    config = DatasetConfig(
        label=label,
        split=split,
        local_dir=local_dir,
        preferred_vendors_only=True,
    )
    fs = SeamlessInteractionFS(config=config, filelist_path=filelist_path)
    print(f"Downloading {file_id} from S3 (one file at a time for consistency)...")
    fs.gather_file_id_data_from_s3(file_id, num_workers=1)

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
    fps: float,
) -> list[tuple[int, int, int]]:
    """
    Apply length rules: drop < 3 s, keep 3–10 s as one segment, break > 10 s into 10 s chunks.
    Uses fps so max duration never exceeds 10 s in wall-clock time (e.g. 29.97 fps -> 299 frames).
    Returns list of (start_frame, end_frame, emotion_index).
    """
    max_frames = min(MAX_SEGMENT_FRAMES, int(fps * 10.0))
    if max_frames < MIN_SEGMENT_FRAMES:
        max_frames = MIN_SEGMENT_FRAMES
    segments = []
    for start, end, em in runs:
        n = end - start
        if n < MIN_SEGMENT_FRAMES:
            continue
        if n <= max_frames:
            segments.append((start, end, em))
            continue
        # Break into max_frames (≤10 s) chunks; remainder ≥ 3 s kept, else dropped
        s = start
        while s < end:
            seg_end = min(s + max_frames, end)
            if seg_end - s >= MIN_SEGMENT_FRAMES:
                segments.append((s, seg_end, em))
            s = seg_end
    return segments


def load_speech_intervals_from_vad(json_path: str) -> list[tuple[float, float]]:
    """
    Load participant JSON and return list of (start, end) in seconds where the
    participant is speaking. Uses metadata:vad; if present, also uses metadata:transcript
    (phrase-level start/end) so we get longer runs when VAD is sparse.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Participant JSON not found: {json_path}. Needed for VAD (speaking-only segments)."
        )
    with open(json_path) as f:
        data = json.load(f)
    intervals = []
    vad = data.get("metadata:vad")
    if vad is not None:
        for entry in vad:
            if "start" not in entry or "end" not in entry:
                continue
            if entry.get("is_speech", True) is True:
                intervals.append((float(entry["start"]), float(entry["end"])))
    transcript = data.get("metadata:transcript")
    if transcript is not None:
        for entry in transcript:
            if "start" in entry and "end" in entry:
                intervals.append((float(entry["start"]), float(entry["end"])))
    if not intervals:
        raise KeyError(
            f"JSON {json_path} has no 'metadata:vad' or 'metadata:transcript' with start/end. "
            "Required to keep only segments where the person is talking (not listening)."
        )
    # #region agent log
    try:
        _dbg = open("/home/benson/Downloads/seamless_interaction/.cursor/debug.log", "a")
        _dbg.write(json.dumps({"id": "vad_load", "timestamp": __import__("time").time() * 1000, "location": "load_speech_intervals_from_vad", "message": "VAD loaded", "data": {"json_path": json_path, "vad_len": len(vad) if vad else 0, "transcript_len": len(transcript) if transcript else 0, "intervals_count": len(intervals), "first_interval": intervals[0] if intervals else None, "last_interval": intervals[-1] if len(intervals) > 1 else None}, "hypothesisId": "C"}) + "\n")
        _dbg.close()
    except Exception:
        pass
    # #endregion
    return intervals


def merge_speech_intervals(
    intervals: list[tuple[float, float]],
    max_gap_sec: float = VAD_MERGE_GAP_SEC,
) -> list[tuple[float, float]]:
    """Merge VAD intervals that are close (gap <= max_gap_sec) into continuous speech runs."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    for s, e in sorted_intervals[1:]:
        if s <= merged[-1][1] + max_gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def segment_has_continuous_speech(
    start_time: float,
    end_time: float,
    merged_runs: list[tuple[float, float]],
    min_continuous_sec: float = MIN_CONTINUOUS_SPEECH_SEC,
) -> bool:
    """True if segment overlaps a continuous-speech run of length >= min_continuous_sec by at least min_continuous_sec."""
    for run_start, run_end in merged_runs:
        run_len = run_end - run_start
        if run_len < min_continuous_sec:
            continue
        overlap_start = max(start_time, run_start)
        overlap_end = min(end_time, run_end)
        overlap_len = overlap_end - overlap_start
        if overlap_len >= min_continuous_sec:
            return True
    return False


def segment_speech_overlap_fraction(
    start_time: float,
    end_time: float,
    speech_intervals: list[tuple[float, float]],
) -> float:
    """Return fraction of [start_time, end_time] covered by speech intervals (0..1)."""
    if start_time >= end_time:
        return 0.0
    duration = end_time - start_time
    total_speech = 0.0
    for s, e in speech_intervals:
        overlap_start = max(start_time, s)
        overlap_end = min(end_time, e)
        if overlap_end > overlap_start:
            total_speech += overlap_end - overlap_start
    return total_speech / duration if duration > 0 else 0.0


def filter_segments_by_speech(
    segments: list[tuple[int, int, int]],
    fps: float,
    speech_intervals: list[tuple[float, float]],
    min_speech_fraction: float = MIN_SPEECH_FRACTION,
) -> list[tuple[int, int, int]]:
    """
    Keep only segments where the participant is talking (VAD overlap >= min_speech_fraction).
    Drops segments that are mostly listening.
    """
    kept = []
    # #region agent log
    _log_path = "/home/benson/Downloads/seamless_interaction/.cursor/debug.log"
    # #endregion
    for i, (start_frame, end_frame, em) in enumerate(segments):
        start_time = start_frame / fps
        end_time = end_frame / fps
        frac = segment_speech_overlap_fraction(start_time, end_time, speech_intervals)
        if frac >= min_speech_fraction:
            kept.append((start_frame, end_frame, em))
        # #region agent log
        if i < 5:
            try:
                _dbg = open(_log_path, "a")
                _dbg.write(json.dumps({"id": f"seg_{i}", "timestamp": __import__("time").time() * 1000, "location": "filter_segments_by_speech", "message": "segment overlap", "data": {"start_time": start_time, "end_time": end_time, "duration": end_time - start_time, "frac": frac, "kept": frac >= min_speech_fraction, "min_required": min_speech_fraction, "speech_intervals_len": len(speech_intervals)}, "hypothesisId": "A"}) + "\n")
                _dbg.close()
            except Exception:
                pass
        # #endregion
    return kept


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


# Precision for ffmpeg time args so video and audio get identical start/duration.
_FFMPEG_TIME_PRECISION = 9


def extract_video_segment_ffmpeg(
    input_mp4: str,
    output_mp4: str,
    start_time: float,
    duration_seconds: float,
) -> None:
    """Extract segment from input_mp4 using exact start_time and duration_seconds (same values as audio for sync).
    Uses -i then -ss (output seek) and re-encode so the segment is exactly [start_time, start_time+duration];
    -ss before -i with -c copy would seek to keyframe and include extra content (e.g. interviewer)."""
    start_s = format(start_time, f".{_FFMPEG_TIME_PRECISION}f")
    duration_s = format(duration_seconds, f".{_FFMPEG_TIME_PRECISION}f")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_mp4,
        "-ss", start_s,
        "-t", duration_s,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-avoid_negative_ts", "1",
        output_mp4,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio_segment_ffmpeg(
    input_wav: str,
    output_wav: str,
    start_time: float,
    duration_seconds: float,
) -> None:
    """Extract segment from input_wav using exact start_time and duration_seconds (same values as video for sync)."""
    start_s = format(start_time, f".{_FFMPEG_TIME_PRECISION}f")
    duration_s = format(duration_seconds, f".{_FFMPEG_TIME_PRECISION}f")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_s,
        "-i", input_wav,
        "-t", duration_s,
        "-acodec", "copy",
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
    duration_seconds: float,
    dominant_emotion_index: int,
) -> None:
    metadata = {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "dominant_emotion_index": int(dominant_emotion_index),
        "dominant_emotion_name": EMOTION_NAMES[int(dominant_emotion_index)],
    }
    path = os.path.join(segment_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def _is_emotion_key(key: str) -> bool:
    """True if this NPZ key is emotion-related (valence, arousal, scores, tokens)."""
    k = key.lower()
    return (
        "emotion" in k
        or "valence" in k
        or "arousal" in k
        or key in ("movement:EmotionArousalToken", "movement:EmotionValenceToken")
    )


def write_segment_emotion_json(
    segment_dir: str,
    npz_path: str,
    start_frame: int,
    end_frame: int,
    dominant_emotion_index: int,
) -> None:
    """
    Write segment_data.json with valence, arousal, ALL emotion labels (emotion_scores:
    Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise), and any
    other emotion-related arrays (e.g. EmotionValenceToken, EmotionArousalToken).
    All per-frame; human-readable.
    """
    data = np.load(npz_path)
    out = {
        "dominant_emotion_index": int(dominant_emotion_index),
        "dominant_emotion_name": EMOTION_NAMES[int(dominant_emotion_index)],
        "emotion_names": EMOTION_NAMES,
    }
    for key in data.files:
        if not _is_emotion_key(key):
            continue
        arr = np.asarray(data[key])
        if arr.ndim < 1 or arr.shape[0] < end_frame:
            continue
        sl = np.squeeze(arr[start_frame:end_frame])
        json_key = key.replace(":", "_")
        out[json_key] = sl.tolist()
    path = os.path.join(segment_dir, "segment_data.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def process_segments(
    base_path: str,
    file_id: str,
    segments: list[tuple[int, int, int]],
    fps: float,
    scores: np.ndarray,
) -> str:
    """
    For each segment: extract video/audio, slice NPZ, write metadata.
    Dominant emotion per segment is argmax(mean(scores[start:end], axis=0)), not the run label.
    Returns path to segments directory.
    """
    mp4_path = os.path.join(base_path, f"{file_id}.mp4")
    wav_path = os.path.join(base_path, f"{file_id}.wav")
    npz_path = os.path.join(base_path, f"{file_id}.npz")
    segments_dir = os.path.join(base_path, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    for idx, (start_frame, end_frame, emotion_index) in enumerate(segments):
        # Segment-level dominant: average emotion_scores over this segment, then argmax.
        segment_scores = scores[start_frame:end_frame]
        segment_dominant = int(np.argmax(np.mean(segment_scores, axis=0)))

        # Single source of truth: one start_time and one duration_seconds for BOTH video and audio.
        # Both extractors receive the exact same numeric values so .wav and .mp4 are 100% synced.
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration_seconds = end_time - start_time
        seg_name = f"segment_{idx:03d}"
        segment_dir = os.path.join(segments_dir, seg_name)
        os.makedirs(segment_dir, exist_ok=True)

        out_mp4 = os.path.join(segment_dir, "video.mp4")
        out_wav = os.path.join(segment_dir, "audio.wav")
        extract_video_segment_ffmpeg(mp4_path, out_mp4, start_time, duration_seconds)
        extract_audio_segment_ffmpeg(wav_path, out_wav, start_time, duration_seconds)
        slice_npz_by_frames(npz_path, start_frame, end_frame, segment_dir)
        write_segment_metadata(
            segment_dir,
            start_frame,
            end_frame,
            start_time,
            end_time,
            duration_seconds,
            segment_dominant,
        )
        write_segment_emotion_json(
            segment_dir, npz_path, start_frame, end_frame, segment_dominant
        )
        print(f"  Wrote {seg_name} ({start_time:.2f}s–{end_time:.2f}s, {EMOTION_NAMES[segment_dominant]})")

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

    allowed_file_ids = load_allowed_file_ids(
        args.interviewee_file_ids,
        args.interviewee_list,
    )
    if allowed_file_ids is not None:
        print(f"Interviewee filter: {len(allowed_file_ids)} file_id(s) from external mapping.")

    if file_id is None:
        file_id = get_first_file_id(filelist_path, require_imitator, allowed_file_ids)
        print(f"Using first subject from filelist: {file_id}")
    elif allowed_file_ids is not None and file_id not in allowed_file_ids:
        raise ValueError(
            f"file_id {file_id} is not in the interviewee list ({len(allowed_file_ids)} ids). "
            "Only file_ids from --interviewee_file_ids/--interviewee_list are allowed when set."
        )

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
    segments = split_runs_into_segments(runs, fps)

    if not segments:
        print("No segments of 3–10 s with uniform dominant emotion found. Exiting.")
        return

    json_path = os.path.join(base_path, f"{file_id}.json")
    speech_intervals = load_speech_intervals_from_vad(json_path)
    min_speech_fraction = args.min_speech_fraction
    n_before = len(segments)
    segments = filter_segments_by_speech(segments, fps, speech_intervals, min_speech_fraction)
    # If threshold is strict and no segments pass, keep segments with any speech at all (exclude only pure listening).
    if not segments and min_speech_fraction > 0:
        runs = find_uniform_emotion_runs(dominant)
        segments_full = split_runs_into_segments(runs, fps)
        segments = filter_segments_by_speech(segments_full, fps, speech_intervals, 0.0)
        if segments:
            print(f"No segments had ≥{min_speech_fraction:.0%} speech; keeping {len(segments)} segment(s) that contain any speech (person talking at least briefly).")
    n_dropped = n_before - len(segments) if segments else n_before
    if n_dropped and segments:
        print(f"Filtered out {n_dropped} segment(s) where participant was talking < {min_speech_fraction:.0%} of the time (kept only segments with ≥{min_speech_fraction:.0%} speech).")

    if not segments:
        print(
            "No segments left after requiring participant to be talking (VAD). "
            "No segment overlapped any speech interval. Exiting."
        )
        return

    # Keep only segments that contain at least min_continuous_speech s of continuous speech (thinking pauses merged; waiting for other person = split).
    min_continuous_speech = args.min_continuous_speech
    vad_merge_gap = args.vad_merge_gap
    merged_runs = merge_speech_intervals(speech_intervals, max_gap_sec=vad_merge_gap)
    n_before_cont = len(segments)
    segments = [
        (sf, ef, em)
        for sf, ef, em in segments
        if segment_has_continuous_speech(sf / fps, ef / fps, merged_runs, min_continuous_speech)
    ]
    if n_before_cont > len(segments):
        print(
            f"Filtered out {n_before_cont - len(segments)} segment(s) without ≥{min_continuous_speech:.1f} s continuous speech (kept only segments with no pauses)."
        )

    if not segments:
        print(
            f"No segments left after requiring ≥{min_continuous_speech:.1f} s continuous speech. Exiting."
        )
        return

    print(f"Found {len(segments)} segments (3–10 s, uniform emotion, ≥{min_continuous_speech:.1f} s continuous speech).")
    segments_dir = process_segments(base_path, file_id, segments, fps, scores)
    delete_large_files(base_path, file_id)
    print("All done. Segments saved under {}; original video/audio/npz removed.".format(segments_dir))


if __name__ == "__main__":
    main()

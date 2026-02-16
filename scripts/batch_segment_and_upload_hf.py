# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch run download_and_segment_participant over the filelist, collect segments
into a HuggingFace dataset (video, audio, all emotion metadata), and push to the
Hub every N participants. Resumable via a progress log.

Usage (from repo root):
  Load .env (HUGGINGFACE_TOKEN, HUGGINGFACE_NAME) then:
  python scripts/batch_segment_and_upload_hf.py
  python scripts/batch_segment_and_upload_hf.py --limit 5 --push_every_n 2
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOCAL_DIR = "/home/benson/Downloads/seamless_segment_dataset"
_PROGRESS_FILENAME = ".batch_segment_progress.json"
_DATASET_CARD_PATH = Path(__file__).resolve().parent / "seamless_segments_dataset_card.md"


def load_dotenv(env_path: Path | None = None) -> None:
    """Set os.environ from .env file (KEY=VALUE lines). No new dependency."""
    if env_path is None:
        # Try repo root first, then cwd (so .env in current dir works too)
        for candidate in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
            if candidate.is_file():
                env_path = candidate
                break
        else:
            return
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip("\ufeff")  # BOM
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1].replace('\\"', '"')
                elif v.startswith("'") and v.endswith("'"):
                    v = v[1:-1].replace("\\'", "'")
                os.environ.setdefault(k, v)
    # Accept common alternative names for HuggingFace credentials
    os.environ.setdefault("HUGGINGFACE_TOKEN", os.environ.get("HF_TOKEN", ""))
    os.environ.setdefault(
        "HUGGINGFACE_NAME",
        os.environ.get("HUGGINGFACE_USERNAME", os.environ.get("HF_USERNAME", os.environ.get("HF_NAME", ""))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch segment filelist participants and upload segments to HuggingFace."
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
        default=_DEFAULT_LOCAL_DIR,
        help="Local root for downloads and segments",
    )
    parser.add_argument(
        "--push_every_n",
        type=int,
        default=2,
        metavar="N",
        help="Push dataset to Hub after every N successful participants",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo id (default: {HUGGINGFACE_NAME}/seamless-segment-dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max number of participants to process (for testing)",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Ignore progress log and process all participants from the start (clears progress for --local_dir)",
    )
    return parser.parse_args()


def progress_log_path(local_dir: str) -> Path:
    return Path(local_dir) / _PROGRESS_FILENAME


def load_progress(local_dir: str) -> dict:
    path = progress_log_path(local_dir)
    if not path.is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def save_progress(local_dir: str, progress: dict) -> None:
    path = progress_log_path(local_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def get_done_file_ids(progress: dict) -> set[str]:
    return {fid for fid, ent in progress.items() if ent.get("status") == "success"}


def run_segment_script(
    file_id: str,
    filelist: str,
    local_dir: str,
    repo_root: Path,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "download_and_segment_participant.py"),
        "--file_id",
        file_id,
        "--filelist",
        filelist,
        "--local_dir",
        local_dir,
        "--min_continuous_speech",
        "3",
        "--vad_merge_gap",
        "2",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))


def collect_segment_rows(
    local_dir: str,
    file_id: str,
    row: pd.Series,
) -> list[dict]:
    """Build one row per segment: video path, audio path, metadata, segment_data, file_id, etc."""
    label = str(row["label"])
    split = str(row["split"])
    batch_idx = int(row["batch_idx"])
    archive_idx = int(row["archive_idx"])
    base = (
        Path(local_dir)
        / label
        / split
        / f"{batch_idx:04d}"
        / f"{archive_idx:04d}"
        / "segments"
    )
    if not base.is_dir():
        return []
    rows = []
    for seg_dir in sorted(base.iterdir()):
        if not seg_dir.is_dir() or not seg_dir.name.startswith("segment_"):
            continue
        video_path = seg_dir / "video.mp4"
        audio_path = seg_dir / "audio.wav"
        meta_path = seg_dir / "metadata.json"
        data_path = seg_dir / "segment_data.json"
        if not video_path.is_file() or not audio_path.is_file():
            continue
        metadata = {}
        if meta_path.is_file():
            with open(meta_path) as f:
                metadata = json.load(f)
        segment_data = {}
        if data_path.is_file():
            with open(data_path) as f:
                segment_data = json.load(f)
        rows.append({
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "file_id": file_id,
            "segment_id": seg_dir.name,
            "label": label,
            "split": split,
            "batch_idx": batch_idx,
            "archive_idx": archive_idx,
            "metadata": metadata,
            "segment_data": segment_data,
        })
    return rows


def _upload_dataset_card(hf_repo: str, token: str) -> None:
    """Upload the dataset card (README.md) to the dataset repo so the Hub displays it."""
    if not _DATASET_CARD_PATH.is_file():
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            path_or_fileobj=str(_DATASET_CARD_PATH),
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="dataset",
        )
    except Exception:
        pass  # Do not fail the push if the card upload fails


def build_and_push_dataset(
    all_rows: list[dict],
    hf_repo: str,
    token: str,
) -> None:
    """Build Dataset from rows (Audio and Video typed for Hub viewer playback) and push to hub."""
    from datasets import Audio, Dataset, Video, concatenate_datasets, load_dataset

    if not all_rows:
        return
    video_bytes = []
    audio_paths = []
    file_ids = []
    segment_ids = []
    labels = []
    splits = []
    batch_idxs = []
    archive_idxs = []
    metadata_strs = []
    segment_data_strs = []
    for r in all_rows:
        with open(r["video_path"], "rb") as f:
            video_bytes.append(f.read())
        audio_paths.append(r["audio_path"])
        file_ids.append(r["file_id"])
        segment_ids.append(r["segment_id"])
        labels.append(r["label"])
        splits.append(r["split"])
        batch_idxs.append(r["batch_idx"])
        archive_idxs.append(r["archive_idx"])
        metadata_strs.append(json.dumps(r["metadata"]))
        segment_data_strs.append(json.dumps(r["segment_data"]))

    d_new = Dataset.from_dict({
        "video": video_bytes,
        "audio": audio_paths,
        "file_id": file_ids,
        "segment_id": segment_ids,
        "label": labels,
        "split": splits,
        "batch_idx": batch_idxs,
        "archive_idx": archive_idxs,
        "metadata": metadata_strs,
        "segment_data": segment_data_strs,
    })
    # Viewer playback: Audio with 48 kHz (dataset WAV rate); Video feature for player
    d_new = d_new.cast_column("audio", Audio(sampling_rate=48000))
    d_new = d_new.cast_column("video", Video())
    try:
        existing = load_dataset(hf_repo, token=token, split="train")
        d = concatenate_datasets([existing, d_new])
    except Exception:
        d = d_new
    d.push_to_hub(hf_repo, token=token, private=False)
    _upload_dataset_card(hf_repo, token)


def load_existing_dataset(hf_repo: str, token: str):
    """Load dataset from hub if it exists."""
    from datasets import load_dataset

    try:
        return load_dataset(hf_repo, token=token, split="train")
    except Exception:
        return None


def main() -> None:
    load_dotenv()
    args = parse_args()
    token = os.environ.get("HUGGINGFACE_TOKEN")
    hf_name = os.environ.get("HUGGINGFACE_NAME")
    if not token or not hf_name:
        print("Set HUGGINGFACE_TOKEN and HUGGINGFACE_NAME in .env (or environment).")
        sys.exit(1)
    hf_repo = args.hf_repo or f"{hf_name}/seamless-segment-dataset"

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    if args.from_scratch:
        progress = {}
        save_progress(local_dir, progress)
        print("From-scratch: progress log cleared.")
    else:
        progress = load_progress(local_dir)
    done = get_done_file_ids(progress)

    if not Path(args.filelist).is_file():
        print(f"Filelist not found: {args.filelist}")
        sys.exit(1)
    df = pd.read_csv(args.filelist)
    df = df[df["has_imitator_movement"] == 1].reset_index(drop=True)
    file_ids = df["file_id"].astype(str).tolist()
    if args.limit is not None:
        file_ids = file_ids[: args.limit]
    todo = [fid for fid in file_ids if fid not in done]
    print(f"Filelist: {len(file_ids)} with has_imitator_movement=1; {len(done)} already done; {len(todo)} to process.")

    rows_since_last_push: list[dict] = []
    success_count_since_push = 0

    for i, file_id in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] Processing {file_id} ...")
        result = run_segment_script(
            file_id, args.filelist, local_dir, _REPO_ROOT
        )
        if result.returncode != 0:
            progress[file_id] = {
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": (result.stderr or result.stdout or "")[:500],
            }
            save_progress(local_dir, progress)
            print(f"  Failed: {result.stderr or result.stdout}")
            continue

        row_series = df[df["file_id"].astype(str) == file_id].iloc[0]
        segment_rows = collect_segment_rows(local_dir, file_id, row_series)
        n_seg = len(segment_rows)
        progress[file_id] = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "segment_count": n_seg,
        }
        save_progress(local_dir, progress)
        rows_since_last_push.extend(segment_rows)
        success_count_since_push += 1
        print(f"  Segments: {n_seg} (rows since last push: {len(rows_since_last_push)})")

        if success_count_since_push >= args.push_every_n:
            print(f"  Pushing to Hub ({hf_repo}) ...")
            build_and_push_dataset(rows_since_last_push, hf_repo, token)
            success_count_since_push = 0
            rows_since_last_push = []
            print("  Push done.")

    if rows_since_last_push:
        print("  Final push to Hub ...")
        build_and_push_dataset(rows_since_last_push, hf_repo, token)
    print("Batch segment and upload complete.")


if __name__ == "__main__":
    main()

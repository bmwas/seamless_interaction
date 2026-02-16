# Seamless Interaction — Emotion Segments

This dataset is a **derived, segmented version** of the [Seamless Interaction Dataset](https://ai.meta.com/research/seamless-interaction/) (Meta). Each row is one short clip (segment) of a single participant with aligned video, audio, and emotion labels.

---

## Source dataset

- **Name:** [Seamless Interaction Dataset](https://ai.meta.com/research/seamless-interaction/)
- **Provider:** Meta Platforms, Inc. (FAIR Seamless Team)
- **Content:** Large-scale face-to-face interactions (video, audio, transcripts, VAD, movement/emotion features from the Imitator model)
- **Original format:** Per-participant files (e.g. one `.mp4`, `.wav`, `.npz`, `.json` per `file_id`) organized by label/split/batch/archive
- **HuggingFace (original):** [facebook/seamless-interaction](https://huggingface.co/datasets/facebook/seamless-interaction)

Only participants with **imitator movement features** (`has_imitator_movement=1` in the filelist) are used.

---

## How this dataset was created

1. **Input:** The official filelist (e.g. `assets/filelist.csv`) is filtered to rows with `has_imitator_movement == 1`. Each such row corresponds to one participant (`file_id`).

2. **Per participant (script: `download_and_segment_participant.py`):**
   - The participant’s data is downloaded from the source (S3) into a local directory.
   - **Emotion runs:** Per-frame dominant emotion is computed as `argmax(movement:emotion_scores)` (8 classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise). Contiguous frames with the same dominant are grouped into runs.
   - **Segmentation:** Runs are split into segments of **3–10 seconds** (by FPS). Only segments where the participant is **speaking** are kept:
     - At least 50% of the segment duration must overlap with speech intervals (from `metadata:vad` and `metadata:transcript`).
     - At least **3 seconds** of **continuous speech** (with up to 2 s gaps merged as “thinking pause”) must fall inside the segment.
   - **Segment label:** For each segment, the **segment-level dominant emotion** is **not** the run label; it is `argmax(mean(emotion_scores[start_frame:end_frame], axis=0))` — i.e. average emotion scores over the segment, then argmax.
   - **Extraction:** For each segment, video and audio are cut with **ffmpeg** (output seek + re-encode for video so the clip is exactly the requested time window; audio copied). Per-segment metadata and emotion-related arrays (valence, arousal, emotion_scores, etc.) are written to the segment folder. Original full-length `.mp4`, `.wav`, and `.npz` are deleted after segmenting; only segment subfolders are kept.

3. **Upload (script: `batch_segment_and_upload_hf.py`):**
   - The batch script iterates over all filelist participants (with the above filter), runs the segment script for each, collects all segment rows (video, audio, metadata, segment_data), and pushes to this HuggingFace dataset repo.
   - **Progress** is logged locally so runs can be resumed (already-processed participants are skipped).
   - The dataset is pushed every N successful participants (e.g. every 2). New segment rows are concatenated with any existing data on the Hub before each push.
   - **Dataset card:** This README is the dataset card; it is uploaded to the repo when the batch script pushes.

**Parameters used in the segmentation (defaults):**

- Segment length: **3–10 s** (by FPS).
- Min speech fraction: **0.5** (50% of segment must be speech).
- Min continuous speech: **3 s** (within segment, with 2 s merge gap for pauses).
- VAD merge gap: **2 s** (gaps ≤ 2 s merged into one “turn”; larger gaps split).

---

## Dataset structure

Each **row** in this dataset is **one segment** (one clip of one participant).

| Column        | Type   | Description |
|---------------|--------|-------------|
| `video`       | Video  | Segment video (MP4), same time window as `audio`. |
| `audio`       | Audio  | Segment audio (WAV), 48 kHz. |
| `file_id`     | string | Original participant ID (e.g. `V00_S0809_I00000309_P0947`). |
| `segment_id`  | string | Segment name (e.g. `segment_000`). |
| `label`       | string | Dataset label: `improvised` or `naturalistic`. |
| `split`       | string | Split: `train`, `dev`, or `test`. |
| `batch_idx`   | int    | Batch index from the original layout. |
| `archive_idx` | int    | Archive index from the original layout. |
| `metadata`    | string | JSON string: `start_frame`, `end_frame`, `start_time`, `end_time`, `duration_seconds`, `dominant_emotion_index`, `dominant_emotion_name`. |
| `segment_data`| string | JSON string: full emotion-related data for the segment (dominant index/name, emotion_scores, valence, arousal, tokens, etc.; see original movement features). |

**Emotion classes (index 0–7):** Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

---

## Intended use

- Training or evaluating models that use short, emotion-labeled audiovisual clips (e.g. emotion recognition, multimodal fusion).
- Studying affect over 3–10 s speaking segments with aligned video, audio, and continuous emotion/valence/arousal.

---

## License and attribution

This derived dataset follows the **license and terms of the original Seamless Interaction Dataset** (e.g. CC-BY-NC 4.0 or as specified by Meta). You must comply with the original dataset’s license and usage policy.

**Attribution:** Segments are derived from the [Seamless Interaction Dataset](https://ai.meta.com/research/seamless-interaction/) by Meta Platforms, Inc. Processing and segmentation were done with the open-source [seamless_interaction](https://github.com/facebookresearch/seamless_interaction) scripts (`download_and_segment_participant.py`, `batch_segment_and_upload_hf.py`).

---

## How to load

```python
from datasets import load_dataset

ds = load_dataset("YOUR_USERNAME/seamless-segment-dataset", split="train")
# Each row: video, audio (48 kHz), file_id, segment_id, label, split, batch_idx, archive_idx, metadata (JSON string), segment_data (JSON string)
```

Parse metadata and segment_data with `json.loads(row["metadata"])` and `json.loads(row["segment_data"])` for numeric and emotion fields.

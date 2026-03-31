#!/usr/bin/env python3
"""
Presenter Detection — Silero VAD + MediaPipe FaceLandmarker

Detects segments where the presenter is actively speaking by combining:
1. Voice Activity Detection (Silero VAD) — finds speech in audio
2. Lip Motion Detection (MediaPipe FaceLandmarker) — confirms lip movement in video

Usage:
    python3 presenter_detection.py \
        --audio /path/to/audio.wav \
        --video /path/to/proxy.mp4 \
        --output /path/to/presenter_segments.json \
        --lip-threshold 0.25 \
        --vad-threshold 0.75 \
        --buffer 0.1
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Logging to stderr (stdout is reserved for JSON output) ──────────

def log(msg: str) -> None:
    print(f"[presenter-detection] {msg}", file=sys.stderr, flush=True)


def log_error(msg: str) -> None:
    print(f"[presenter-detection] ERROR: {msg}", file=sys.stderr, flush=True)


# ── Step 1: Voice Activity Detection ────────────────────────────────

def read_wav_as_tensor(audio_path: str, target_sr: int = 16000):
    """Read a WAV file and return a torch tensor at the target sample rate."""
    import torch
    import wave
    import struct

    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f'<{n_frames * n_channels}h'
        samples = struct.unpack(fmt, raw_data)
        audio_np = np.array(samples, dtype=np.float32) / 32768.0
    elif sampwidth == 4:
        fmt = f'<{n_frames * n_channels}i'
        samples = struct.unpack(fmt, raw_data)
        audio_np = np.array(samples, dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert to mono if stereo
    if n_channels > 1:
        audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

    # Simple resample if needed (linear interpolation)
    if framerate != target_sr:
        duration = len(audio_np) / framerate
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio_np) - 1, target_len)
        audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np).astype(np.float32)

    return torch.from_numpy(audio_np)


def run_vad(audio_path: str, vad_threshold: float) -> list[dict]:
    """Run Silero VAD on audio file, return list of {start, end} in seconds."""
    import torch
    torch.set_num_threads(1)

    from silero_vad import load_silero_vad, get_speech_timestamps

    log(f"Loading Silero VAD model...")
    model = load_silero_vad()

    log(f"Reading audio: {audio_path}")
    audio = read_wav_as_tensor(audio_path, target_sr=16000)

    log(f"Running VAD (threshold={vad_threshold})...")
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        threshold=vad_threshold,
        sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )

    segments = []
    for ts in speech_timestamps:
        start_sec = ts["start"] / 16000.0
        end_sec = ts["end"] / 16000.0
        segments.append({"start": round(start_sec, 3), "end": round(end_sec, 3)})

    log(f"VAD found {len(segments)} speech segments")
    return segments


# ── Step 2: Lip Motion Detection ────────────────────────────────────

def get_model_path() -> str:
    """Get path to the FaceLandmarker model file."""
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "face_landmarker_v2_with_blendshapes.task"
    if not model_path.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {model_path}. "
            f"Run setup.sh to download it."
        )
    return str(model_path)


def compute_lip_distance(landmarks) -> tuple[float, float]:
    """
    Compute normalized lip distance from face landmarks.

    Returns (normalized_lip_distance, face_width).
    Landmarks 13 = inner upper lip, 14 = inner lower lip.
    Landmarks 234 = left face edge, 454 = right face edge.
    """
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    lip_dist = math.sqrt(
        (upper_lip.x - lower_lip.x) ** 2
        + (upper_lip.y - lower_lip.y) ** 2
    )

    left_face = landmarks[234]
    right_face = landmarks[454]
    face_width = math.sqrt(
        (left_face.x - right_face.x) ** 2
        + (left_face.y - right_face.y) ** 2
    )

    if face_width < 1e-6:
        return 0.0, 0.0

    normalized = lip_dist / face_width
    return normalized, face_width


def get_face_center(landmarks) -> tuple[float, float]:
    """Get center of face (nose tip, landmark 1)."""
    nose = landmarks[1]
    return nose.x, nose.y


def get_face_area(landmarks) -> float:
    """Estimate face bounding box area from landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width * height


def run_lip_detection(
    video_path: str,
    vad_segments: list[dict],
    lip_threshold: float,
) -> list[dict]:
    """
    For each VAD segment, check lip motion in the video frames.
    Returns segments with confidence scores based on lip movement.
    """
    import mediapipe as mp

    model_path = get_model_path()
    log(f"Loading FaceLandmarker model from: {model_path}")

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=3,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )

    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    frame_duration = 1.0 / fps

    log(f"Video FPS: {fps:.1f}, analyzing lip motion in {len(vad_segments)} segments")

    # Sliding window size for lip variance detection
    VARIANCE_WINDOW = 5
    # Standard deviation threshold: above = active speech, below = static open mouth
    VARIANCE_THRESHOLD = 0.02

    results = []

    for seg_idx, segment in enumerate(vad_segments):
        start_time = segment["start"]
        end_time = segment["end"]

        # Seek to start frame
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        lip_moving_frames = 0
        total_frames = 0
        faces_detected_frames = 0
        # Reset face center tracking per segment (we seek to a new position)
        prev_face_center = None
        # Sliding window of recent lip distances for variance calculation
        lip_distance_window: list[float] = []

        log(f"  Segment {seg_idx}: {start_time:.2f}-{end_time:.2f}s (frames {start_frame}-{end_frame})")

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            timestamp_ms = int(frame_num * frame_duration * 1000)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                log(f"  Frame {frame_num}: detection error: {e}")
                continue

            if not detection_result.face_landmarks:
                continue

            faces_detected_frames += 1

            # Find the presenter — face with the largest bounding box area
            best_face_idx = 0
            best_area = 0.0
            for i, face_landmarks in enumerate(detection_result.face_landmarks):
                area = get_face_area(face_landmarks)
                if area > best_area:
                    best_area = area
                    best_face_idx = i

            presenter_landmarks = detection_result.face_landmarks[best_face_idx]
            normalized_lip, face_width = compute_lip_distance(presenter_landmarks)

            # Body motion filtering: if the whole face moved significantly
            # between frames, it's body/camera motion, not speech
            current_center = get_face_center(presenter_landmarks)
            is_body_motion = False
            if prev_face_center is not None:
                center_delta = math.sqrt(
                    (current_center[0] - prev_face_center[0]) ** 2
                    + (current_center[1] - prev_face_center[1]) ** 2
                )
                if face_width > 0 and center_delta > 0.20 * face_width:
                    is_body_motion = True

            prev_face_center = current_center

            if is_body_motion:
                continue

            # Add current lip distance to sliding window
            lip_distance_window.append(normalized_lip)
            if len(lip_distance_window) > VARIANCE_WINDOW:
                lip_distance_window.pop(0)

            # Variance-based speech detection:
            # Instead of checking if lip distance > threshold (catches static open mouth),
            # check if lip distances VARY over recent frames (actual speech = lips moving)
            if len(lip_distance_window) >= VARIANCE_WINDOW:
                std_dev = float(np.std(lip_distance_window))
                if std_dev > VARIANCE_THRESHOLD:
                    lip_moving_frames += 1

        # Calculate confidence: ratio of lip-moving frames to total frames
        confidence = lip_moving_frames / total_frames if total_frames > 0 else 0.0

        log(f"    -> faces_detected={faces_detected_frames}/{total_frames}, "
            f"lip_moving={lip_moving_frames}/{total_frames}, "
            f"confidence={confidence:.3f}")

        results.append({
            "start": start_time,
            "end": end_time,
            "confidence": round(confidence, 3),
            "lip_frames": lip_moving_frames,
            "total_frames": total_frames,
        })

        if (seg_idx + 1) % 10 == 0 or seg_idx == len(vad_segments) - 1:
            log(f"  Processed segment {seg_idx + 1}/{len(vad_segments)}")

    cap.release()
    detector.close()

    log(f"Lip detection complete for {len(results)} segments")
    return results


# ── Step 3: Merge and finalize segments ─────────────────────────────

def merge_segments(
    segments: list[dict],
    buffer: float,
    video_duration: float,
) -> list[dict]:
    """
    Apply buffer, merge nearby segments, and filter short ones.

    1. Add buffer to each side of each segment
    2. Merge segments closer than 0.5s
    3. Filter out segments shorter than 0.3s
    """
    if not segments:
        return []

    # Apply buffer and clamp to video bounds
    buffered = []
    for seg in segments:
        start = max(0.0, seg["start"] - buffer)
        end = min(video_duration, seg["end"] + buffer) if video_duration > 0 else seg["end"] + buffer
        confidence = seg["confidence"]
        buffered.append({"start": round(start, 3), "end": round(end, 3), "confidence": confidence})

    # Sort by start time
    buffered.sort(key=lambda s: s["start"])

    # Merge segments closer than 0.5s
    merged = [buffered[0]]
    for seg in buffered[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] < 0.5:
            # Merge: extend end, average confidence weighted by duration
            prev_dur = prev["end"] - prev["start"]
            seg_dur = seg["end"] - seg["start"]
            total_dur = prev_dur + seg_dur
            if total_dur > 0:
                prev["confidence"] = round(
                    (prev["confidence"] * prev_dur + seg["confidence"] * seg_dur) / total_dur,
                    3,
                )
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg)

    # Filter out segments shorter than 0.3s
    filtered = [s for s in merged if (s["end"] - s["start"]) >= 0.3]

    return filtered


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Presenter detection using Silero VAD + MediaPipe FaceLandmarker")
    parser.add_argument("--audio", required=True, help="Path to audio.wav (16kHz mono)")
    parser.add_argument("--video", required=True, help="Path to proxy.mp4")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    parser.add_argument("--lip-threshold", type=float, default=0.25, help="Normalized lip distance threshold (default: 0.25)")
    parser.add_argument("--vad-threshold", type=float, default=0.75, help="VAD speech probability threshold (default: 0.75)")
    parser.add_argument("--buffer", type=float, default=0.1, help="Buffer in seconds to add around segments (default: 0.1)")
    args = parser.parse_args()

    start_time = time.time()

    # Validate inputs
    if not os.path.isfile(args.audio):
        log_error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    if not os.path.isfile(args.video):
        log_error(f"Video file not found: {args.video}")
        sys.exit(1)

    video_duration = get_video_duration(args.video)
    log(f"Video duration: {video_duration:.1f}s")

    # Step 1: VAD
    log("=== Step 1: Voice Activity Detection ===")
    vad_segments = run_vad(args.audio, args.vad_threshold)

    if not vad_segments:
        log("No speech detected by VAD. Returning empty result.")
        result = {
            "segments": [],
            "total_speech_duration": 0.0,
            "total_video_duration": round(video_duration, 3),
            "speech_ratio": 0.0,
            "processing_time_ms": round((time.time() - start_time) * 1000),
        }
        output_json = json.dumps(result, ensure_ascii=False, indent=2)
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(output_json)
        return

    # Step 2: Lip motion detection
    log("=== Step 2: Lip Motion Detection ===")
    face_detection_failed = False
    try:
        segments_with_confidence = run_lip_detection(
            args.video, vad_segments, args.lip_threshold
        )
    except FileNotFoundError as e:
        log_error(str(e))
        log("Falling back to VAD-only segments (no face model)")
        face_detection_failed = True
        segments_with_confidence = [
            {"start": s["start"], "end": s["end"], "confidence": 1.0}
            for s in vad_segments
        ]
    except Exception as e:
        log_error(f"Lip detection failed: {e}")
        log("Falling back to VAD-only segments")
        face_detection_failed = True
        segments_with_confidence = [
            {"start": s["start"], "end": s["end"], "confidence": 1.0}
            for s in vad_segments
        ]

    # Step 2b: Filter by confidence
    log("=== Step 2b: Filtering segments by confidence ===")
    pre_filter_count = len(segments_with_confidence)

    # Fix 1: Discard segments where lips moved in ≤15% of frames (likely not the presenter)
    # Fix 2: Discard long segments (>10s) with low confidence (<0.3) — likely production assistant talking
    filtered_segments = []
    for seg in segments_with_confidence:
        seg_duration = seg["end"] - seg["start"]
        conf = seg["confidence"]

        if conf <= 0.18:
            log(f"    Filtered out segment {seg['start']:.2f}-{seg['end']:.2f}s "
                f"(confidence {conf:.3f} <= 0.18)")
            continue

        if seg_duration > 10 and conf < 0.3:
            log(f"    Filtered out long segment {seg['start']:.2f}-{seg['end']:.2f}s "
                f"({seg_duration:.1f}s, confidence {conf:.3f} < 0.3)")
            continue

        filtered_segments.append(seg)

    log(f"  Confidence filter: {pre_filter_count} -> {len(filtered_segments)} segments "
        f"({pre_filter_count - len(filtered_segments)} removed)")

    # Step 3: Merge and finalize
    log("=== Step 3: Merging segments ===")
    final_segments = merge_segments(filtered_segments, args.buffer, video_duration)

    # Calculate totals
    total_speech = sum(s["end"] - s["start"] for s in final_segments)
    speech_ratio = total_speech / video_duration if video_duration > 0 else 0.0

    # VAD-only fallback: if lip detection filtered too aggressively
    if speech_ratio < 0.15 and video_duration > 30 and not face_detection_failed:
        log(f"Low speech ratio ({speech_ratio:.2f}), falling back to VAD-only mode")

        # Re-build segments from original VAD output (before lip filtering)
        vad_only_segments = [
            {"start": s["start"], "end": s["end"], "confidence": 0.5}
            for s in vad_segments
        ]

        # Apply same merge/filter logic
        final_segments = merge_segments(vad_only_segments, args.buffer, video_duration)

        # Recalculate totals
        total_speech = sum(s["end"] - s["start"] for s in final_segments)
        speech_ratio = total_speech / video_duration if video_duration > 0 else 0.0

        log(f"VAD-only: {len(final_segments)} segments, "
            f"{total_speech:.1f}s speech ({speech_ratio:.2f})")

    # Clean output — only start, end, confidence
    output_segments = [
        {"start": s["start"], "end": s["end"], "confidence": s["confidence"]}
        for s in final_segments
    ]

    processing_time_ms = round((time.time() - start_time) * 1000)

    result = {
        "segments": output_segments,
        "total_speech_duration": round(total_speech, 3),
        "total_video_duration": round(video_duration, 3),
        "speech_ratio": round(speech_ratio, 3),
        "processing_time_ms": processing_time_ms,
    }

    log(f"Final: {len(output_segments)} segments, {total_speech:.1f}s speech / {video_duration:.1f}s total ({speech_ratio:.0%})")
    log(f"Processing time: {processing_time_ms}ms")

    # Write to output file
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    Path(args.output).write_text(output_json, encoding="utf-8")

    # Also print to stdout
    print(output_json)


if __name__ == "__main__":
    main()

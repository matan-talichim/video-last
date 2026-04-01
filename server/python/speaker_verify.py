#!/usr/bin/env python3
"""
Speaker Verification via WeSpeaker — cosine similarity against a reference embedding.

Builds a reference embedding from the top-N highest-confidence VAD segments
(most likely the presenter), then scores every word against that reference.

Used by merge_transcript.py when --speaker-verify is passed.
Logs to stderr (stdout reserved for JSON).
"""

import sys
import time

import numpy as np

_model = None


def _get_model():
    """Lazy-load WeSpeaker ONNX model (loaded once, reused)."""
    global _model
    if _model is None:
        import wespeaker
        _model = wespeaker.load_model("english")
        print("[speaker-verify] WeSpeaker model loaded", file=sys.stderr)
    return _model


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def build_reference_embedding(audio_path, segments, top_n=5):
    """
    Build a reference speaker embedding from the top-N highest-confidence
    VAD segments (these are most likely the presenter).

    Args:
        audio_path: Path to WAV file.
        segments: List of dicts with 'start', 'end', 'confidence'.
        top_n: Number of top segments to average.

    Returns:
        numpy array — averaged reference embedding, or None if extraction fails.
    """
    start_time = time.time()
    model = _get_model()

    # Sort by confidence descending, take top N
    best = sorted(segments, key=lambda s: s.get("confidence", 0), reverse=True)[:top_n]

    if not best:
        print("[speaker-verify] No segments for reference embedding", file=sys.stderr)
        return None

    embeddings = []
    for seg in best:
        seg_start = seg["start"]
        seg_end = seg["end"]
        duration = seg_end - seg_start

        # Skip very short segments — embedding quality is poor under 0.3s
        if duration < 0.3:
            continue

        try:
            emb = model.extract_embedding_wav(audio_path, seg_start, seg_end)
            if emb is not None and len(emb) > 0:
                embeddings.append(np.array(emb).flatten())
        except Exception as e:
            print(f"[speaker-verify] Embedding extraction failed for segment "
                  f"{seg_start:.2f}-{seg_end:.2f}: {e}", file=sys.stderr)

    if not embeddings:
        print("[speaker-verify] No valid embeddings extracted", file=sys.stderr)
        return None

    reference = np.mean(embeddings, axis=0)
    elapsed = int((time.time() - start_time) * 1000)
    print(f"[speaker-verify] Reference embedding built from {len(embeddings)} segments, "
          f"{elapsed}ms", file=sys.stderr)
    return reference


def verify_speaker(audio_path, words, reference_embedding, threshold_high=0.6, threshold_low=0.4):
    """
    Score each word against the reference embedding.

    For each word, extracts an embedding and computes cosine similarity.
    Words shorter than 0.15s are grouped with adjacent words to improve
    embedding quality.

    Args:
        audio_path: Path to WAV file.
        words: List of word dicts (must have 'start', 'end').
        reference_embedding: Numpy array from build_reference_embedding().
        threshold_high: Above this → is_presenter = True (override).
        threshold_low: Below this → is_presenter = False (override).

    Returns:
        (words, stats) — words updated with 'speaker_score', stats dict.
    """
    start_time = time.time()
    model = _get_model()

    promoted = 0
    demoted = 0
    scored = 0
    skipped = 0

    # Group words into chunks for better embedding quality.
    # Individual words are often too short for reliable embeddings.
    chunks = _build_chunks(words, min_duration=0.4, max_duration=2.0)

    for chunk in chunks:
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        duration = chunk_end - chunk_start

        if duration < 0.15:
            # Too short even as a chunk — skip scoring
            for w in chunk:
                w["speaker_score"] = -1.0
            skipped += len(chunk)
            continue

        try:
            emb = model.extract_embedding_wav(audio_path, chunk_start, chunk_end)
            if emb is None or len(emb) == 0:
                for w in chunk:
                    w["speaker_score"] = -1.0
                skipped += len(chunk)
                continue

            emb = np.array(emb).flatten()
            similarity = _cosine_similarity(emb, reference_embedding)

            for w in chunk:
                w["speaker_score"] = round(similarity, 4)
                old_val = w["is_presenter"]

                if similarity > threshold_high and not old_val:
                    w["is_presenter"] = True
                    promoted += 1
                elif similarity < threshold_low and old_val:
                    w["is_presenter"] = False
                    demoted += 1

            scored += len(chunk)

        except Exception as e:
            print(f"[speaker-verify] Scoring failed for chunk "
                  f"{chunk_start:.2f}-{chunk_end:.2f}: {e}", file=sys.stderr)
            for w in chunk:
                w["speaker_score"] = -1.0
            skipped += len(chunk)

    elapsed = int((time.time() - start_time) * 1000)
    stats = {
        "scored_words": scored,
        "skipped_words": skipped,
        "promoted": promoted,
        "demoted": demoted,
        "processing_time_ms": elapsed,
    }

    print(
        f"[speaker-verify] Done: scored={scored}, skipped={skipped}, "
        f"promoted={promoted}, demoted={demoted}, {elapsed}ms",
        file=sys.stderr,
    )
    return words, stats


def _build_chunks(words, min_duration=0.4, max_duration=2.0):
    """
    Group consecutive words into chunks that are at least min_duration long.
    Keeps chunks under max_duration. Respects gaps > 0.3s as chunk boundaries.
    """
    chunks = []
    current_chunk = []

    for w in words:
        if not current_chunk:
            current_chunk.append(w)
            continue

        gap = w["start"] - current_chunk[-1]["end"]
        chunk_duration = w["end"] - current_chunk[0]["start"]

        # Start new chunk if: big gap, or chunk would be too long
        if gap > 0.3 or chunk_duration > max_duration:
            chunks.append(current_chunk)
            current_chunk = [w]
        else:
            current_chunk.append(w)

    if current_chunk:
        chunks.append(current_chunk)

    # Merge very short chunks with neighbors
    merged = []
    for chunk in chunks:
        duration = chunk[-1]["end"] - chunk[0]["start"]
        if merged and duration < min_duration:
            prev = merged[-1]
            prev_duration = prev[-1]["end"] - prev[0]["start"]
            gap = chunk[0]["start"] - prev[-1]["end"]
            if gap < 0.3 and prev_duration + duration <= max_duration:
                merged[-1] = prev + chunk
                continue
        merged.append(chunk)

    return merged

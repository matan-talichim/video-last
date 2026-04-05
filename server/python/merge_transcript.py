#!/usr/bin/env python3
"""
Merge transcript words with presenter segments — word-level output.

For each word in transcript.json, checks overlap with presenter_segments.json.
Words with >= 60% overlap with a presenter segment (+ buffer) are kept as presenter words.

Output: JSON with flat word array (each word has id, word, start, end, is_presenter, confidence).
Logs to stderr.
"""

import argparse
import json
import sys
import time
import wave

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge transcript with presenter segments")
    parser.add_argument("--transcript", required=True, help="Path to transcript.json")
    parser.add_argument("--segments", required=True, help="Path to presenter_segments.json")
    parser.add_argument("--output", required=True, help="Path to output merged_transcript.json")
    parser.add_argument("--buffer", type=float, default=0.25, help="Buffer in seconds around each segment")
    parser.add_argument("--audio", default=None, help="Path to audio.wav for RMS volume filtering")
    parser.add_argument("--speaker-verify", action="store_true", default=False,
                        help="Enable WeSpeaker speaker verification (requires wespeaker-onnx)")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_overlap(word_start, word_end, seg_start, seg_end):
    """Compute the overlap duration between a word and a segment."""
    overlap_start = max(word_start, seg_start)
    overlap_end = min(word_end, seg_end)
    return max(0.0, overlap_end - overlap_start)


def load_audio(path):
    """Load a WAV file and return (samples_as_float64, sample_rate)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    # If stereo, take first channel
    if n_channels > 1:
        samples = samples[::n_channels]

    # Normalize to [-1, 1]
    max_val = float(2 ** (sampwidth * 8 - 1))
    samples = samples / max_val

    return samples, sample_rate


def compute_word_rms(audio_data, sample_rate, start, end):
    """Compute RMS for a time range in the audio."""
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    start_sample = max(0, min(start_sample, len(audio_data)))
    end_sample = max(start_sample, min(end_sample, len(audio_data)))
    if end_sample <= start_sample:
        return 0.0
    segment = audio_data[start_sample:end_sample]
    return float(np.sqrt(np.mean(segment ** 2)))


def is_presenter_word(word, segments, buffer):
    """Check if at least 60% of the word duration overlaps with any buffered segment."""
    w_start = word["start"]
    w_end = word["end"]
    w_duration = w_end - w_start

    if w_duration <= 0:
        # Zero-duration word: check if it falls inside any segment
        for seg in segments:
            if (seg["start"] - buffer) <= w_start <= (seg["end"] + buffer):
                return True
        return False

    total_overlap = 0.0
    for seg in segments:
        seg_start = seg["start"] - buffer
        seg_end = seg["end"] + buffer
        total_overlap += compute_overlap(w_start, w_end, seg_start, seg_end)

    return (total_overlap / w_duration) >= 0.55


def main():
    args = parse_args()
    start_time = time.time()

    # Load inputs
    transcript = load_json(args.transcript)
    words = transcript.get("words", [])

    # Load segments (fallback: treat all words as presenter)
    try:
        segments_data = load_json(args.segments)
        segments = segments_data.get("segments", [])
    except (FileNotFoundError, json.JSONDecodeError):
        print("[merge] No valid presenter_segments found, treating all words as presenter", file=sys.stderr)
        segments = []

    # Build flat word array with sequential IDs
    word_list = []

    for idx, w in enumerate(words):
        entry = {
            "id": idx,
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "confidence": w.get("confidence", 0.0),
        }
        word_list.append(entry)

    # ── Compute raw signals ──

    # RMS per word
    if args.audio:
        try:
            audio_data, sample_rate = load_audio(args.audio)
            for entry in word_list:
                entry["rms"] = compute_word_rms(audio_data, sample_rate, entry["start"], entry["end"])
            print(f"[merge] RMS computed for {len(word_list)} words", file=sys.stderr)
        except Exception as e:
            print(f"[merge] RMS computation failed, skipping: {e}", file=sys.stderr)

    # Speaker verification (WeSpeaker) — adds speaker_score per word
    speaker_verify_stats = None
    if args.speaker_verify and args.audio:
        try:
            from speaker_verify import build_reference_embedding, verify_speaker

            print("[merge] Running WeSpeaker speaker verification...", file=sys.stderr)

            reference = build_reference_embedding(args.audio, segments)

            if reference is not None:
                # verify_speaker adds speaker_score to each word
                # We pass is_presenter=True for all temporarily — word_scorer will decide
                for entry in word_list:
                    entry["is_presenter"] = True
                word_list, speaker_verify_stats = verify_speaker(
                    args.audio, word_list, reference,
                    threshold_high=0.6, threshold_low=0.4,
                )
                print(
                    f"[merge] Speaker scores computed "
                    f"(promoted={speaker_verify_stats['promoted']}, "
                    f"demoted={speaker_verify_stats['demoted']})",
                    file=sys.stderr,
                )
            else:
                print("[merge] Speaker verification skipped: no reference embedding", file=sys.stderr)
        except ImportError as e:
            print(f"[merge] Speaker verification unavailable (missing dependency): {e}", file=sys.stderr)
        except Exception as e:
            print(f"[merge] Speaker verification failed, skipping: {e}", file=sys.stderr)
    elif args.speaker_verify and not args.audio:
        print("[merge] Speaker verification skipped: --audio required", file=sys.stderr)

    # ── Centralized scoring — single decision per word ──
    from word_scorer import score_all_words

    word_list = score_all_words(word_list, segments)

    # Before sorting: assign take_id based on gaps in ORIGINAL order
    take_id = 0
    for i in range(len(word_list)):
        word_list[i]['take_id'] = take_id
        if i < len(word_list) - 1:
            gap = word_list[i + 1]['start'] - word_list[i]['end']
            if gap > 0.5:
                take_id += 1

    print(f"[merge] Assigned {take_id + 1} takes before chronological sort", file=sys.stderr)

    # Sort words chronologically and re-assign IDs
    word_list.sort(key=lambda w: w['start'])
    for i, w in enumerate(word_list):
        w['id'] = i

    # Map final_decision to is_presenter
    for entry in word_list:
        entry["is_presenter"] = entry["final_decision"] != "reject"

    presenter_count = sum(1 for w in word_list if w["is_presenter"])
    other_count = len(word_list) - presenter_count

    processing_time_ms = int((time.time() - start_time) * 1000)

    result = {
        "words": word_list,
        "stats": {
            "total_words": len(words),
            "presenter_words": presenter_count,
            "other_words": other_count,
            "filter_ratio": round(presenter_count / len(words), 2) if words else 1.0,
            "processing_time_ms": processing_time_ms,
        },
    }

    if speaker_verify_stats:
        result["stats"]["speaker_verify"] = speaker_verify_stats

    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write to stdout
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)

    print(f"\n[merge] Done: {presenter_count}/{len(words)} presenter words, {processing_time_ms}ms", file=sys.stderr)


if __name__ == "__main__":
    main()

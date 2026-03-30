#!/usr/bin/env python3
"""
Merge transcript words with presenter segments — word-level output.

For each word in transcript.json, checks overlap with presenter_segments.json.
Words with >= 50% overlap with a presenter segment (+ buffer) are kept as presenter words.

Output: JSON with flat word array (each word has id, word, start, end, is_presenter, confidence).
Logs to stderr.
"""

import argparse
import json
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Merge transcript with presenter segments")
    parser.add_argument("--transcript", required=True, help="Path to transcript.json")
    parser.add_argument("--segments", required=True, help="Path to presenter_segments.json")
    parser.add_argument("--output", required=True, help="Path to output merged_transcript.json")
    parser.add_argument("--buffer", type=float, default=0.25, help="Buffer in seconds around each segment")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_overlap(word_start, word_end, seg_start, seg_end):
    """Compute the overlap duration between a word and a segment."""
    overlap_start = max(word_start, seg_start)
    overlap_end = min(word_end, seg_end)
    return max(0.0, overlap_end - overlap_start)


def is_presenter_word(word, segments, buffer):
    """Check if at least 50% of the word duration overlaps with any buffered segment."""
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

    return (total_overlap / w_duration) >= 0.5


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
    presenter_count = 0
    other_count = 0

    for idx, w in enumerate(words):
        is_pres = is_presenter_word(w, segments, args.buffer) if segments else True
        word_list.append({
            "id": idx,
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "is_presenter": is_pres,
            "confidence": w.get("confidence", 0.0),
        })
        if is_pres:
            presenter_count += 1
        else:
            other_count += 1

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

    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write to stdout
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)

    print(f"\n[merge] Done: {presenter_count}/{len(words)} presenter words, {processing_time_ms}ms", file=sys.stderr)


if __name__ == "__main__":
    main()

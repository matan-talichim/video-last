#!/usr/bin/env python3
"""
Merge transcript words with presenter segments.

For each word in transcript.json, checks overlap with presenter_segments.json.
Words with >= 50% overlap with a presenter segment (+ buffer) are kept as presenter words.

Output: JSON to stdout, logs to stderr.
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


def build_utterances(presenter_words):
    """Build contiguous utterances from presenter words.
    Words with a gap > 0.5s start a new utterance."""
    if not presenter_words:
        return []

    utterances = []
    current_words = [presenter_words[0]]

    for word in presenter_words[1:]:
        prev_end = current_words[-1]["end"]
        if word["start"] - prev_end > 0.5:
            # Flush current utterance
            text = " ".join(w["word"] for w in current_words)
            utterances.append({
                "text": text,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
            })
            current_words = [word]
        else:
            current_words.append(word)

    # Flush last utterance
    if current_words:
        text = " ".join(w["word"] for w in current_words)
        utterances.append({
            "text": text,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
        })

    return utterances


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

    # If no segments, all words are presenter
    if not segments:
        presenter_words = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "confidence": w.get("confidence", 0.0)}
            for w in words
        ]
        other_words = []
    else:
        presenter_words = []
        other_words = []
        for w in words:
            entry = {
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "confidence": w.get("confidence", 0.0),
            }
            if is_presenter_word(w, segments, args.buffer):
                presenter_words.append(entry)
            else:
                entry["speaker"] = "other"
                other_words.append(entry)

    presenter_text = " ".join(w["word"] for w in presenter_words)
    presenter_utterances = build_utterances(presenter_words)

    processing_time_ms = int((time.time() - start_time) * 1000)

    total = len(words)
    result = {
        "presenter_words": presenter_words,
        "other_words": other_words,
        "presenter_text": presenter_text,
        "presenter_utterances": presenter_utterances,
        "stats": {
            "total_words": total,
            "presenter_words": len(presenter_words),
            "other_words": len(other_words),
            "filter_ratio": round(len(presenter_words) / total, 2) if total > 0 else 1.0,
            "processing_time_ms": processing_time_ms,
        },
    }

    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write to stdout
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)

    print(f"\n[merge] Done: {len(presenter_words)}/{total} presenter words, {processing_time_ms}ms", file=sys.stderr)


if __name__ == "__main__":
    main()

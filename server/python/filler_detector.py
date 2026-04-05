#!/usr/bin/env python3
"""
Filler Word Detector — marks Hebrew filler words for removal.

Pipeline position: AFTER take_selector, BEFORE sentence_builder.

This module identifies filler words in the merged transcript and marks them
with `is_filler: true`. The sentence_builder can then exclude or downweight
these words when building sentences.

Rules:
- A word is a filler ONLY if it stands alone (not part of a meaningful phrase)
- "בעצם" at sentence start = filler → mark for removal
- "בעצם" mid-sentence in meaningful context = NOT filler → keep
- Protection: do NOT mark as filler if removal would create a gap > 500ms
  inside a sentence (the gap check happens at the word level)

Usage:
  python3 filler_detector.py \
    --merged input/$JOB/merged_transcript.json \
    --output input/$JOB/merged_transcript.json

Output: same merged_transcript.json with added `is_filler` field per word.
"""

import argparse
import json
import sys
import time


# ── Filler word lists ──────────────────────────────

# Always fillers when standalone
ALWAYS_FILLERS = {
    "אה", "אהה", "אהם", "אמ", "אממ", "אמממ",
    "נו", "ככה",
}

# Fillers only at sentence start (before a meaningful word)
START_FILLERS = {
    "בעצם", "אוקיי", "אז", "סתם", "רגע", "יאללה",
}

# Multi-word fillers (detected as sequence)
MULTI_WORD_FILLERS = [
    ["משהו", "כזה"],
    ["או", "משהו"],
    ["כזה", "כזה"],
]

# Context words that make a "filler" into a meaningful word
# e.g., "בעצם זה עובד" → "בעצם" is filler; "מה שבעצם קורה" → not filler
CONTEXT_PROTECTORS = {
    "בעצם": ["מה", "שמה", "ש", "כש", "איך"],  # If preceded by these → not filler
    "אז": ["ו", "ואז"],  # "ואז" is a connector, not a filler
    "רגע": ["שנייה", "רק"],  # "רק רגע" might be meaningful
}

# Maximum gap (seconds) that filler removal can create within a sentence
MAX_GAP_AFTER_REMOVAL = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Detect filler words in merged transcript")
    parser.add_argument("--merged", required=True, help="Path to merged_transcript.json")
    parser.add_argument("--output", required=True, help="Path to output (can be same file)")
    return parser.parse_args()


def is_sentence_start(words, idx):
    """Check if word at idx is at the start of a 'sentence' (after a gap > 0.8s or first word)."""
    if idx == 0:
        return True
    prev = words[idx - 1]
    curr = words[idx]
    gap = curr["start"] - prev["end"]
    return gap > 0.8


def get_prev_word_text(words, idx):
    """Get the text of the previous word, or empty string if none."""
    if idx <= 0:
        return ""
    return words[idx - 1].get("word", "").strip()


def would_create_large_gap(words, idx):
    """
    Check if removing word at idx would create a gap > MAX_GAP_AFTER_REMOVAL
    between the previous and next words.
    """
    prev_end = words[idx - 1]["end"] if idx > 0 else None
    next_start = words[idx + 1]["start"] if idx < len(words) - 1 else None

    if prev_end is None or next_start is None:
        return False  # Edge words can be removed safely

    gap = next_start - prev_end
    return gap > MAX_GAP_AFTER_REMOVAL


def detect_fillers(words):
    """
    Mark filler words in the word list.

    Adds `is_filler: true/false` to each word.
    Returns the number of fillers detected.
    """
    filler_count = 0
    n = len(words)

    for i in range(n):
        word = words[i]
        text = word.get("word", "").strip()
        word["is_filler"] = False

        # Skip non-presenter words — they're already excluded
        if word.get("final_decision") == "reject":
            continue

        # Rule 1: Always-filler words (standalone hesitations)
        if text in ALWAYS_FILLERS:
            # Protection: don't remove if it creates a large gap
            if not would_create_large_gap(words, i):
                word["is_filler"] = True
                filler_count += 1
                continue

        # Rule 2: Start-filler words (only at sentence beginning)
        if text in START_FILLERS:
            if is_sentence_start(words, i):
                # Check context protectors
                prev_text = get_prev_word_text(words, i)
                protectors = CONTEXT_PROTECTORS.get(text, [])
                if prev_text not in protectors:
                    if not would_create_large_gap(words, i):
                        word["is_filler"] = True
                        filler_count += 1
                        continue

        # Rule 3: Multi-word fillers
        for mwf in MULTI_WORD_FILLERS:
            if i + len(mwf) <= n:
                match = True
                for j, filler_word in enumerate(mwf):
                    if words[i + j].get("word", "").strip() != filler_word:
                        match = False
                        break
                if match:
                    # Mark all words in the multi-word filler
                    for j in range(len(mwf)):
                        if not would_create_large_gap(words, i + j):
                            words[i + j]["is_filler"] = True
                            filler_count += 1

    return filler_count


def main():
    args = parse_args()
    start_time = time.time()

    # Load
    with open(args.merged, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]

    # Detect
    filler_count = detect_fillers(words)

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Add stats
    if "stats" not in data:
        data["stats"] = {}
    data["stats"]["filler_detection"] = {
        "fillers_detected": filler_count,
        "processing_time_ms": processing_time_ms,
    }

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[filler_detector] Done: {filler_count} fillers detected in {len(words)} words, {processing_time_ms}ms",
          file=sys.stderr)

    # Also output to stdout for pipeline
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
WhisperX forced alignment — improve word-level timestamps.

Takes audio.wav + transcript words (with Deepgram timestamps),
runs WhisperX forced alignment, and returns words with corrected start/end.

Fallback: if WhisperX fails, returns the original Deepgram timestamps unchanged.

CLI:
    python3 align_words.py \
        --audio audio.wav \
        --words merged_transcript.json \
        --output aligned_transcript.json

Logs to stderr.
"""

import argparse
import json
import sys
import time


def log(msg):
    print(f"[align_words] {msg}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="WhisperX forced alignment for word timestamps")
    parser.add_argument("--audio", required=True, help="Path to audio.wav")
    parser.add_argument("--words", required=True, help="Path to merged_transcript.json")
    parser.add_argument("--output", required=True, help="Path to output aligned_transcript.json")
    parser.add_argument("--language", default="he", help="Language code (default: he for Hebrew)")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_with_whisperx(audio_path, words, language):
    """
    Run WhisperX forced alignment on audio using the transcript words.

    Returns a dict mapping word index → (new_start, new_end), or None on failure.
    """
    try:
        import whisperx
        import torch
    except ImportError as e:
        log(f"WhisperX not available ({e}), using original timestamps")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading WhisperX alignment model (device={device}, language={language})")

        # Build transcript segments for WhisperX alignment
        # Group words into segments by gaps > 1.0s
        segments = []
        current_segment_words = []

        for w in words:
            if current_segment_words:
                gap = w["start"] - current_segment_words[-1]["end"]
                if gap > 1.0:
                    # Flush current segment
                    seg_text = " ".join(cw["word"] for cw in current_segment_words)
                    segments.append({
                        "start": current_segment_words[0]["start"],
                        "end": current_segment_words[-1]["end"],
                        "text": seg_text,
                    })
                    current_segment_words = []

            current_segment_words.append(w)

        # Flush last segment
        if current_segment_words:
            seg_text = " ".join(cw["word"] for cw in current_segment_words)
            segments.append({
                "start": current_segment_words[0]["start"],
                "end": current_segment_words[-1]["end"],
                "text": seg_text,
            })

        if not segments:
            log("No segments to align")
            return None

        log(f"Aligning {len(segments)} segments, {len(words)} words")

        # Load alignment model and align
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result = whisperx.align(
            segments, model_a, metadata, audio_path, device,
            return_char_alignments=False,
        )

        # Extract aligned words from result
        aligned_words = []
        for seg in result.get("segments", []):
            for aw in seg.get("words", []):
                if "start" in aw and "end" in aw:
                    aligned_words.append(aw)

        if not aligned_words:
            log("WhisperX returned no aligned words")
            return None

        log(f"WhisperX returned {len(aligned_words)} aligned words")

        # Match aligned words back to original words by text similarity and position
        alignments = {}
        aligned_idx = 0

        for orig_idx, orig_word in enumerate(words):
            orig_text = orig_word["word"].strip()

            # Find best matching aligned word near the expected position
            best_match = None
            best_dist = float("inf")

            search_start = max(0, aligned_idx - 3)
            search_end = min(len(aligned_words), aligned_idx + 5)

            for ai in range(search_start, search_end):
                aw = aligned_words[ai]
                aw_text = aw.get("word", "").strip()
                if aw_text == orig_text:
                    dist = abs(ai - aligned_idx)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = ai

            if best_match is not None:
                aw = aligned_words[best_match]
                alignments[orig_idx] = (aw["start"], aw["end"])
                aligned_idx = best_match + 1

        log(f"Matched {len(alignments)}/{len(words)} words to WhisperX alignment")
        return alignments

    except Exception as e:
        log(f"WhisperX alignment failed: {e}")
        return None


def main():
    args = parse_args()
    start_time = time.time()

    # Load merged transcript
    data = load_json(args.words)
    words = data.get("words", [])

    if not words:
        log("No words to align")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        return

    # Try WhisperX alignment
    alignments = align_with_whisperx(args.audio, words, args.language)

    updated_count = 0
    if alignments:
        for idx, word in enumerate(words):
            if idx in alignments:
                new_start, new_end = alignments[idx]
                # Sanity check: new timestamps should be close to originals (within 0.5s)
                orig_start = word["start"]
                orig_end = word["end"]
                if abs(new_start - orig_start) < 0.5 and abs(new_end - orig_end) < 0.5:
                    word["start"] = round(new_start, 3)
                    word["end"] = round(new_end, 3)
                    updated_count += 1

        log(f"Updated {updated_count}/{len(words)} word timestamps via WhisperX")
    else:
        log("Using original Deepgram timestamps (WhisperX fallback)")

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Add alignment stats
    data["stats"]["alignment"] = {
        "method": "whisperx" if updated_count > 0 else "deepgram_original",
        "updated_words": updated_count,
        "total_words": len(words),
        "processing_time_ms": processing_time_ms,
    }

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Also write to stdout
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)

    log(f"Done: {updated_count}/{len(words)} words aligned, {processing_time_ms}ms")


if __name__ == "__main__":
    main()

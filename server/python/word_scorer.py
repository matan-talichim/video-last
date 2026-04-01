#!/usr/bin/env python3
"""
Centralized word scoring — collects all signals per word
and makes a single presenter/reject decision.

Each word receives 5 scores from different sources:
  - vad_score (0-1): does this word fall within a VAD speech segment?
  - visual_score (0-1): presenter_segments confidence at this time
  - speaker_score (0-1): WeSpeaker similarity to presenter
  - asr_score (0-1): Deepgram confidence for this word
  - energy_score (0-1): RMS normalized to reference

A weighted final_score determines the decision:
  - >= 0.6: presenter
  - < 0.4: reject
  - 0.4-0.6: uncertain (AI decides)

Logs to stderr.
"""

import sys

# ── Weights ─────────────────────────────────────────
WEIGHT_SPEAKER = 0.35
WEIGHT_VAD = 0.20
WEIGHT_ENERGY = 0.20
WEIGHT_ASR = 0.15
WEIGHT_VISUAL = 0.10

THRESHOLD_PRESENTER = 0.50
THRESHOLD_REJECT = 0.35

# Energy normalization cap (very loud)
ENERGY_CAP = 0.3


def log(msg):
    print(f"[word_scorer] {msg}", file=sys.stderr)


def compute_vad_score(word, presenter_segments):
    """
    1.0 if word fully inside a presenter segment.
    0.5-0.9 if partially overlapping.
    0.0 if no overlap at all.
    """
    best_overlap = 0.0
    word_dur = word["end"] - word["start"]
    if word_dur <= 0:
        return 0.0

    for seg in presenter_segments:
        overlap_start = max(word["start"], seg["start"])
        overlap_end = min(word["end"], seg["end"])
        overlap = max(0, overlap_end - overlap_start)
        ratio = overlap / word_dur
        best_overlap = max(best_overlap, ratio)

    return best_overlap


def compute_visual_score(word, presenter_segments):
    """
    Returns the confidence of the presenter segment that contains this word.
    0.0 if word is not in any segment.
    """
    for seg in presenter_segments:
        if seg["start"] <= word["start"] and word["end"] <= seg["end"]:
            return seg.get("confidence", 0.5)
    return 0.0


def compute_speaker_score(word):
    """
    WeSpeaker similarity — already computed by speaker_verify.
    If not available, returns 0.5 (neutral).
    """
    return word.get("speaker_score", 0.5)


def compute_energy_score(word):
    """
    RMS of this word normalized to reference.
    Already computed in merge_transcript — just normalize to 0-1.
    """
    rms = word.get("rms", 0.0)
    if rms <= 0:
        return 0.0
    return min(rms / ENERGY_CAP, 1.0)


def compute_final_score(word):
    """
    Weighted combination of all scores.
    """
    return (
        word.get("speaker_score_norm", 0.5) * WEIGHT_SPEAKER
        + word.get("vad_score", 0.0) * WEIGHT_VAD
        + word.get("energy_score", 0.0) * WEIGHT_ENERGY
        + word.get("asr_score", 0.5) * WEIGHT_ASR
        + word.get("visual_score", 0.0) * WEIGHT_VISUAL
    )


def make_decision(word):
    """
    Three-tier decision with speaker_score override:
    - speaker_score > 0.60: PRESENTER (strong speaker match alone is enough)
    - speaker_score < 0.30: REJECT (clearly not the presenter)
    - Otherwise use weighted final_score with thresholds 0.50 / 0.35
    """
    speaker = word.get("speaker_score", word.get("speaker_score_norm", 0.5))

    if speaker > 0.60:
        return "presenter"
    if speaker < 0.30:
        return "reject"

    final = word["final_score"]
    if final >= THRESHOLD_PRESENTER:
        return "presenter"
    elif final < THRESHOLD_REJECT:
        return "reject"
    else:
        return "uncertain"


def score_all_words(words, presenter_segments):
    """
    For each word, compute all available scores and make final decision.

    Input: words from transcript with start/end/confidence (+ optional rms, speaker_score)
    Output: same words with added score fields + final_score + final_decision
    """
    for word in words:
        # 1. VAD Score — is there speech at this timestamp?
        word["vad_score"] = compute_vad_score(word, presenter_segments)

        # 2. Visual Score — presenter_segments confidence
        word["visual_score"] = compute_visual_score(word, presenter_segments)

        # 3. Speaker Score — already in word from speaker_verify pass
        word["speaker_score_norm"] = compute_speaker_score(word)

        # 4. ASR Score — Deepgram confidence (already exists)
        word["asr_score"] = word.get("confidence", 0.5)

        # 5. Energy Score — RMS normalized
        word["energy_score"] = compute_energy_score(word)

        # FINAL DECISION
        word["final_score"] = round(compute_final_score(word), 4)
        word["final_decision"] = make_decision(word)

    # Summary stats
    total = len(words)
    if total > 0:
        presenter_count = sum(1 for w in words if w["final_decision"] == "presenter")
        reject_count = sum(1 for w in words if w["final_decision"] == "reject")
        uncertain_count = sum(1 for w in words if w["final_decision"] == "uncertain")

        avg_speaker = sum(w.get("speaker_score_norm", 0.5) for w in words) / total
        avg_vad = sum(w.get("vad_score", 0.0) for w in words) / total
        avg_energy = sum(w.get("energy_score", 0.0) for w in words) / total

        log(f"Scored {total} words:")
        log(f"  presenter: {presenter_count} ({presenter_count/total*100:.1f}%)")
        log(f"  reject: {reject_count} ({reject_count/total*100:.1f}%)")
        log(f"  uncertain: {uncertain_count} ({uncertain_count/total*100:.1f}%)")
        log(f"Score distribution:")
        log(f"  speaker_score avg: {avg_speaker:.2f}")
        log(f"  vad_score avg: {avg_vad:.2f}")
        log(f"  energy_score avg: {avg_energy:.2f}")
        log(
            f"Weights: speaker={int(WEIGHT_SPEAKER*100)}%, "
            f"vad={int(WEIGHT_VAD*100)}%, energy={int(WEIGHT_ENERGY*100)}%, "
            f"asr={int(WEIGHT_ASR*100)}%, visual={int(WEIGHT_VISUAL*100)}%"
        )

    return words


if __name__ == "__main__":
    # Self-test with dummy data
    test_words = [
        {"word": "שלום", "start": 0.0, "end": 0.5, "confidence": 0.95, "rms": 0.15, "speaker_score": 0.8},
        {"word": "עולם", "start": 0.5, "end": 1.0, "confidence": 0.90, "rms": 0.12, "speaker_score": 0.75},
        {"word": "רעש", "start": 2.0, "end": 2.3, "confidence": 0.40, "rms": 0.02, "speaker_score": 0.2},
        {"word": "גבולי", "start": 3.0, "end": 3.5, "confidence": 0.70, "rms": 0.08, "speaker_score": 0.5},
    ]
    test_segments = [
        {"start": 0.0, "end": 1.5, "confidence": 0.9},
    ]

    result = score_all_words(test_words, test_segments)
    for w in result:
        print(
            f"  {w['word']:>6}  final={w['final_score']:.3f}  decision={w['final_decision']}",
            file=sys.stderr,
        )
    log("Self-test passed.")

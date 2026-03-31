#!/usr/bin/env python3
"""
Take Selector — Rule-based duplicate take detection (Agent 2).

Identifies duplicate takes, false starts, production cues, micro-stutters,
and abandoned takes using deterministic rules (no AI).

CLI:
    python3 take_selector.py \
        --words merged_transcript.json \
        --audio audio.wav \
        --video-type general \
        --output take_decisions.json

Output: JSON with remove_ids, decisions, stats.
Logs to stderr.
"""

import argparse
import json
import math
import sys
import time
import wave
from collections import Counter

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


# ── Constants ────────────────────────────────────────

PRODUCTION_CUES = [
    "נתחיל שוב", "עוד פעם", "לא טוב", "רגע רגע",
    "שוב מנקודה זו", "עוצר", "פסול", "סליחה",
    "אחת שתיים", "אני עושה שוב", "בואו נעשה",
    "ממשיכים", "שוב", "טייק", "נתחיל", "מהתחלה",
    "שקט מצלמים", "מוכן", "מוכנים",
]

HEBREW_FILLERS = [
    "אה", "אהה", "אההה", "אמ", "אמם", "אממ",
    "כאילו", "נו", "אוקיי", "בעצם", "סתם",
]

# Thresholds (defaults, overridable via config)
SIMILARITY_THRESHOLD = 0.85
LOOKBACK_SECONDS = 15.0
SCORING_OVERRIDE_MARGIN = 0.20
TARGET_WPS = 2.7


# ── CLI ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rule-based duplicate take detection (Agent 2)"
    )
    parser.add_argument("--words", required=True, help="Path to merged_transcript.json")
    parser.add_argument("--audio", default=None, help="Path to audio.wav for RMS")
    parser.add_argument("--video-type", default="general", help="Video type for aggressiveness")
    parser.add_argument("--output", required=True, help="Path to output take_decisions.json")
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--lookback-seconds", type=float, default=LOOKBACK_SECONDS)
    parser.add_argument("--scoring-override-margin", type=float, default=SCORING_OVERRIDE_MARGIN)
    return parser.parse_args()


# ── Audio helpers ────────────────────────────────────

def load_audio(path):
    """Load WAV file → (float64 samples, sample_rate)."""
    if np is None:
        log("numpy not available, skipping audio loading")
        return None, None

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    if n_channels > 1:
        samples = samples[::n_channels]

    max_val = float(2 ** (sampwidth * 8 - 1))
    samples = samples / max_val

    return samples, sample_rate


def compute_rms(audio_data, sample_rate, start, end):
    """Compute RMS for a time range."""
    if audio_data is None or np is None:
        return 0.0
    s = max(0, int(start * sample_rate))
    e = min(len(audio_data), int(end * sample_rate))
    if e <= s:
        return 0.0
    segment = audio_data[s:e]
    return float(np.sqrt(np.mean(segment ** 2)))


# ── Text helpers ─────────────────────────────────────

def cosine_similarity_bow(words_a, words_b):
    """Cosine similarity between two word lists (bag of words)."""
    counter_a = Counter(words_a)
    counter_b = Counter(words_b)
    all_keys = set(counter_a) | set(counter_b)
    dot = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def ends_with_punctuation(word_text):
    """Check if word ends with sentence-ending punctuation."""
    return word_text.rstrip().endswith((".", "!", "?", "׃"))


def get_gap_after(word, next_word):
    """Gap in ms between end of word and start of next word."""
    if next_word is None:
        return 9999.0
    return (next_word["start"] - word["end"]) * 1000.0


# ── Detection algorithms ────────────────────────────

def detect_micro_stutters(presenter_words):
    """
    Section 4.3: Same string (1-3 chars), appears 2+ times consecutive,
    gap < 200ms. Keep only last occurrence.
    """
    decisions = []
    remove_ids = set()
    i = 0

    while i < len(presenter_words):
        w = presenter_words[i]
        text = w["word"].strip()

        # Only short words (1-3 chars)
        if len(text) > 3:
            i += 1
            continue

        # Find consecutive identical words with small gap
        group = [w]
        j = i + 1
        while j < len(presenter_words):
            nw = presenter_words[j]
            gap_ms = (nw["start"] - group[-1]["end"]) * 1000.0
            if nw["word"].strip() == text and gap_ms < 200:
                group.append(nw)
                j += 1
            else:
                break

        if len(group) >= 2:
            # Keep only last occurrence
            to_remove = [g["id"] for g in group[:-1]]
            remove_ids.update(to_remove)
            decisions.append({
                "ids": to_remove,
                "reason": "stutter",
                "kept_ids": [group[-1]["id"]],
            })

        i = j if len(group) >= 2 else i + 1

    return remove_ids, decisions


def detect_production_cues(presenter_words):
    """
    Section 4.4: Detect production cue phrases.
    Remove cue words + words within 1000ms before and after.
    """
    decisions = []
    remove_ids = set()

    # Build text windows to match multi-word cues
    for i, w in enumerate(presenter_words):
        for cue in PRODUCTION_CUES:
            cue_words = cue.split()
            cue_len = len(cue_words)

            if i + cue_len > len(presenter_words):
                continue

            window = presenter_words[i:i + cue_len]
            window_text = [ww["word"].strip() for ww in window]

            if window_text == cue_words:
                # Found a cue — collect IDs of the cue words
                cue_start = window[0]["start"]
                cue_end = window[-1]["end"]
                cue_ids = [ww["id"] for ww in window]

                # Find words within 1000ms before and after
                tail_ids = []
                for pw in presenter_words:
                    if pw["id"] in cue_ids:
                        continue
                    pw_mid = (pw["start"] + pw["end"]) / 2.0
                    if (cue_start - 1.0) <= pw_mid <= (cue_end + 1.0):
                        tail_ids.append(pw["id"])

                all_ids = cue_ids + tail_ids
                remove_ids.update(all_ids)
                decisions.append({
                    "ids": all_ids,
                    "reason": "production_cue",
                })

    return remove_ids, decisions


def detect_full_abandoned_takes(presenter_words, cue_remove_ids):
    """
    Section 4.5: If a production cue is preceded by silence > 2500ms,
    look backwards to previous major silence, compare text before/after.
    If similar → delete everything from prev silence to cue.
    """
    decisions = []
    remove_ids = set()

    # Find indices where production cues start
    cue_start_indices = []
    for i, w in enumerate(presenter_words):
        if w["id"] in cue_remove_ids:
            # Check if this is the first word of a cue group
            if i == 0 or presenter_words[i - 1]["id"] not in cue_remove_ids:
                cue_start_indices.append(i)

    for cue_idx in cue_start_indices:
        if cue_idx == 0:
            continue

        prev_word = presenter_words[cue_idx - 1]
        cue_word = presenter_words[cue_idx]
        gap_before = (cue_word["start"] - prev_word["end"]) * 1000.0

        if gap_before < 2500:
            continue

        # Look backwards for previous major silence (> 2000ms gap)
        prev_silence_idx = 0
        for k in range(cue_idx - 1, 0, -1):
            gap = (presenter_words[k]["start"] - presenter_words[k - 1]["end"]) * 1000.0
            if gap > 2000:
                prev_silence_idx = k
                break

        # Find end of cue (last consecutive cue ID)
        cue_end_idx = cue_idx
        while cue_end_idx + 1 < len(presenter_words) and \
                presenter_words[cue_end_idx + 1]["id"] in cue_remove_ids:
            cue_end_idx += 1

        # Words before cue (the abandoned take)
        abandoned_words = presenter_words[prev_silence_idx:cue_idx]
        abandoned_words = [w for w in abandoned_words if w["id"] not in cue_remove_ids]

        # Words after cue (the new take)
        after_start = cue_end_idx + 1
        after_end = min(after_start + len(abandoned_words) + 5, len(presenter_words))
        new_take_words = presenter_words[after_start:after_end]
        new_take_words = [w for w in new_take_words if w["id"] not in cue_remove_ids]

        if not abandoned_words or not new_take_words:
            continue

        # Compare text similarity
        abandoned_text = [w["word"].strip() for w in abandoned_words]
        new_text = [w["word"].strip() for w in new_take_words[:len(abandoned_text) + 3]]
        sim = cosine_similarity_bow(abandoned_text, new_text)

        if sim > 0.5:
            # Delete the abandoned take
            ids_to_remove = [w["id"] for w in abandoned_words]
            remove_ids.update(ids_to_remove)
            decisions.append({
                "ids": ids_to_remove,
                "reason": "abandoned_take",
                "kept_ids": [w["id"] for w in new_take_words],
            })

    return remove_ids, decisions


def detect_false_starts(presenter_words, already_removed):
    """
    Section 4.2: sentence < 4 words, no punctuation, gap > 600ms after,
    first 2 words match next sentence.
    """
    decisions = []
    remove_ids = set()

    # Split into sentences by gaps > 600ms
    sentences = []
    current = []
    for i, w in enumerate(presenter_words):
        if w["id"] in already_removed:
            continue
        if current:
            gap_ms = (w["start"] - current[-1]["end"]) * 1000.0
            if gap_ms > 600:
                sentences.append(current)
                current = [w]
            else:
                current.append(w)
        else:
            current = [w]
    if current:
        sentences.append(current)

    for i in range(len(sentences) - 1):
        sent = sentences[i]
        next_sent = sentences[i + 1]

        if len(sent) >= 4:
            continue

        # Check no punctuation at end
        last_word = sent[-1]["word"].strip()
        if ends_with_punctuation(last_word):
            continue

        # Check gap after
        gap_ms = (next_sent[0]["start"] - sent[-1]["end"]) * 1000.0
        if gap_ms < 600:
            continue

        # Check first 2 words match
        if len(sent) >= 2 and len(next_sent) >= 2:
            s_words = [w["word"].strip() for w in sent[:2]]
            n_words = [w["word"].strip() for w in next_sent[:2]]
            if s_words == n_words:
                ids = [w["id"] for w in sent]
                remove_ids.update(ids)
                decisions.append({
                    "ids": ids,
                    "reason": "false_start",
                    "kept_ids": [w["id"] for w in next_sent],
                })

    return remove_ids, decisions


def detect_repetitions(presenter_words, already_removed, threshold, lookback_sec):
    """
    Section 4.1: Sliding window 4-gram comparison.
    Cosine > threshold within lookback window.
    First occurrence = duplicate (remove), last = keep.
    """
    decisions = []
    remove_ids = set()

    # Filter to active words
    active = [w for w in presenter_words if w["id"] not in already_removed]

    if len(active) < 4:
        return remove_ids, decisions

    # Build 4-grams with their word objects
    grams = []
    for i in range(len(active) - 3):
        gram_words = active[i:i + 4]
        gram_text = [w["word"].strip() for w in gram_words]
        grams.append({
            "text": gram_text,
            "words": gram_words,
            "start": gram_words[0]["start"],
            "end": gram_words[-1]["end"],
            "ids": [w["id"] for w in gram_words],
        })

    # Track which grams are part of duplicate groups
    # group_id → list of gram indices
    duplicate_groups = []
    matched = set()

    for i in range(len(grams)):
        if i in matched:
            continue

        group = [i]
        for j in range(i + 1, len(grams)):
            if j in matched:
                continue

            # Check time window
            time_diff = grams[j]["start"] - grams[i]["end"]
            if time_diff > lookback_sec:
                break
            if time_diff < 0:
                continue

            sim = cosine_similarity_bow(grams[i]["text"], grams[j]["text"])
            if sim >= threshold:
                group.append(j)
                matched.add(j)

        if len(group) >= 2:
            matched.add(i)
            duplicate_groups.append(group)

    # For each group: expand grams into full take regions, keep last
    for group in duplicate_groups:
        # Collect all word IDs per take
        takes = []
        for gram_idx in group:
            gram = grams[gram_idx]
            takes.append({
                "ids": set(gram["ids"]),
                "words": list(gram["words"]),
                "start": gram["start"],
                "end": gram["end"],
            })

        # Merge overlapping takes
        merged_takes = []
        for take in takes:
            merged = False
            for mt in merged_takes:
                if take["ids"] & mt["ids"]:
                    mt["ids"] |= take["ids"]
                    mt["words"] = list({w["id"]: w for w in mt["words"] + take["words"]}.values())
                    mt["start"] = min(mt["start"], take["start"])
                    mt["end"] = max(mt["end"], take["end"])
                    merged = True
                    break
            if not merged:
                merged_takes.append(take)

        if len(merged_takes) < 2:
            continue

        # Sort by time — last wins
        merged_takes.sort(key=lambda t: t["start"])
        kept = merged_takes[-1]
        for earlier in merged_takes[:-1]:
            ids = sorted(earlier["ids"] - kept["ids"] - already_removed)
            if ids:
                remove_ids.update(ids)
                decisions.append({
                    "ids": ids,
                    "reason": "duplicate_take",
                    "kept_ids": sorted(kept["ids"]),
                })

    return remove_ids, decisions


# ── Take Scoring ─────────────────────────────────────

def count_fillers(words):
    """Count filler words in a take."""
    return sum(1 for w in words if w["word"].strip() in HEBREW_FILLERS)


def count_stutters_in_take(words):
    """Count stutter-like repetitions in a take."""
    count = 0
    for i in range(1, len(words)):
        if words[i]["word"].strip() == words[i - 1]["word"].strip():
            count += 1
    return count


def score_take(take_words, audio_data=None, sample_rate=None):
    """
    Section 4.6: Score a take.
    Disfluency 40%, ASR confidence 30%, vocal energy 20%, pacing 10%.
    """
    if not take_words:
        return 0.0

    n = len(take_words)

    # 1. Disfluencies (40%)
    fillers = count_fillers(take_words)
    stutters = count_stutters_in_take(take_words)
    disfluency_score = 1.0 - (fillers + stutters) / max(n, 1)
    disfluency_score = max(0.0, disfluency_score)

    # 2. ASR confidence (30%)
    confidences = [w.get("confidence", 0.5) for w in take_words]
    confidence_score = sum(confidences) / len(confidences) if confidences else 0.5

    # 3. Vocal energy (20%)
    if audio_data is not None and sample_rate is not None:
        rms_values = []
        for w in take_words:
            rms = w.get("rms")
            if rms is None:
                rms = compute_rms(audio_data, sample_rate, w["start"], w["end"])
            rms_values.append(rms)
        rms_values = sorted(rms_values)
        energy_score = rms_values[len(rms_values) // 2] if rms_values else 0.0
        # Normalize energy to 0-1 range (cap at 0.3 RMS as max)
        energy_score = min(energy_score / 0.3, 1.0)
    else:
        energy_score = 0.5  # neutral when no audio

    # 4. Pacing (10%) — closest to TARGET_WPS
    duration = take_words[-1]["end"] - take_words[0]["start"]
    if duration > 0:
        wps = n / duration
        pacing_score = 1.0 - min(abs(wps - TARGET_WPS) / TARGET_WPS, 1.0)
    else:
        pacing_score = 0.0

    return (disfluency_score * 0.4 +
            confidence_score * 0.3 +
            energy_score * 0.2 +
            pacing_score * 0.1)


def apply_take_scoring(decisions, presenter_words, audio_data, sample_rate, override_margin):
    """
    Section 4.6: For duplicate_take decisions, score both takes.
    Default: last take wins. Override only if earlier take scores >margin higher.
    """
    word_map = {w["id"]: w for w in presenter_words}
    updated_decisions = []

    for dec in decisions:
        if dec["reason"] != "duplicate_take" or "kept_ids" not in dec:
            updated_decisions.append(dec)
            continue

        removed_words = [word_map[wid] for wid in dec["ids"] if wid in word_map]
        kept_words = [word_map[wid] for wid in dec["kept_ids"] if wid in word_map]

        if not removed_words or not kept_words:
            updated_decisions.append(dec)
            continue

        score_removed = score_take(removed_words, audio_data, sample_rate)
        score_kept = score_take(kept_words, audio_data, sample_rate)

        log(f"Take scoring: removed (score {score_removed:.2f}) vs kept (score {score_kept:.2f})")

        # Override: if earlier take scores significantly higher, swap
        if score_removed > score_kept + override_margin:
            log(f"  → Override! Keeping earlier take (score {score_removed:.2f} > {score_kept:.2f} + {override_margin})")
            updated_decisions.append({
                "ids": dec["kept_ids"],
                "reason": "duplicate_take",
                "kept_ids": dec["ids"],
            })
        else:
            updated_decisions.append(dec)

    return updated_decisions


# ── Logging ──────────────────────────────────────────

def log(msg):
    print(f"[take_selector] {msg}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────

def main():
    args = parse_args()
    start_time = time.time()

    # Load words
    with open(args.words, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", data) if isinstance(data, dict) else data
    if isinstance(words, dict):
        words = words.get("words", [])

    # Filter to presenter words only
    presenter_words = [w for w in words if w.get("is_presenter", True)]
    presenter_words.sort(key=lambda w: w["start"])

    log(f"Analyzing {len(presenter_words)} presenter words...")

    # Load audio if available
    audio_data = None
    sample_rate = None
    if args.audio:
        try:
            audio_data, sample_rate = load_audio(args.audio)
            log(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
        except Exception as e:
            log(f"Could not load audio: {e}")

    all_remove_ids = set()
    all_decisions = []

    # 1. Micro-stutters (4.3)
    stutter_ids, stutter_decs = detect_micro_stutters(presenter_words)
    all_remove_ids |= stutter_ids
    all_decisions.extend(stutter_decs)
    log(f"Micro-stutters: {len(stutter_decs)} groups, {len(stutter_ids)} words")

    # 2. Production cues (4.4)
    cue_ids, cue_decs = detect_production_cues(presenter_words)
    all_remove_ids |= cue_ids
    all_decisions.extend(cue_decs)
    log(f"Production cues: {len(cue_decs)} found, {len(cue_ids)} words")

    # 3. Full abandoned takes (4.5)
    abandoned_ids, abandoned_decs = detect_full_abandoned_takes(presenter_words, all_remove_ids)
    all_remove_ids |= abandoned_ids
    all_decisions.extend(abandoned_decs)
    log(f"Abandoned takes: {len(abandoned_decs)} found, {len(abandoned_ids)} words")

    # 4. False starts (4.2)
    false_start_ids, false_start_decs = detect_false_starts(presenter_words, all_remove_ids)
    all_remove_ids |= false_start_ids
    all_decisions.extend(false_start_decs)
    log(f"False starts: {len(false_start_decs)} found, {len(false_start_ids)} words")

    # 5. Repetition detection (4.1)
    rep_ids, rep_decs = detect_repetitions(
        presenter_words, all_remove_ids,
        args.similarity_threshold, args.lookback_seconds,
    )
    all_remove_ids |= rep_ids
    all_decisions.extend(rep_decs)
    log(f"Repetitions: {len(rep_decs)} groups, {len(rep_ids)} words")

    # 6. Take scoring (4.6) — refine duplicate decisions
    all_decisions = apply_take_scoring(
        all_decisions, presenter_words,
        audio_data, sample_rate,
        args.scoring_override_margin,
    )

    # Recompute remove_ids after scoring may have swapped takes
    all_remove_ids = set()
    for dec in all_decisions:
        all_remove_ids.update(dec["ids"])

    # Stats
    reason_counts = {}
    for dec in all_decisions:
        r = dec["reason"]
        reason_counts[r] = reason_counts.get(r, 0) + 1

    stats = {
        "duplicates_found": reason_counts.get("duplicate_take", 0),
        "false_starts": reason_counts.get("false_start", 0),
        "production_cues": reason_counts.get("production_cue", 0),
        "stutters": reason_counts.get("stutter", 0),
        "abandoned_takes": reason_counts.get("abandoned_take", 0),
        "total_removed_words": len(all_remove_ids),
    }

    log(f"Found {stats['duplicates_found']} duplicate groups, "
        f"{stats['false_starts']} false starts, "
        f"{stats['production_cues']} production cues, "
        f"{stats['stutters']} stutters, "
        f"{stats['abandoned_takes']} abandoned takes")
    log(f"Total: {len(all_remove_ids)} words marked for removal")

    processing_time_ms = int((time.time() - start_time) * 1000)

    result = {
        "remove_ids": sorted(all_remove_ids),
        "decisions": all_decisions,
        "stats": stats,
        "processing_time_ms": processing_time_ms,
    }

    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write to stdout
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)

    log(f"Done in {processing_time_ms}ms")


if __name__ == "__main__":
    main()

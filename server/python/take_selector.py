#!/usr/bin/env python3
"""
Take Selector — Rule-based duplicate take detection (Agent 2).

Identifies duplicate takes, false starts, production cues, micro-stutters,
and abandoned takes using deterministic rules (no AI).

RULE PRIORITY (when rules conflict):

  CRITICAL — never override:
    1. Production Cues → always remove (+ 1000ms tail)
    2. Full Abandoned Take → always remove entire section
    3. Coarticulation (gap < 20ms) → NEVER cut

  IMPORTANT — apply unless conflicts with CRITICAL:
    4. Repetition (cosine > 0.85) → remove first, keep last
    5. False Start (< 4 words + restart) → remove
    6. Micro-stutter → keep last occurrence only

  QUALITY — apply after CRITICAL and IMPORTANT:
    7. Take Scoring → override "last wins" only if earlier scores 20%+ higher
    8. Flawless Run → if nothing to remove, return all words as keep

CONFLICT RESOLUTION:
  - If a word is marked by both Production Cue AND Repetition → Production Cue wins
  - If Take Scoring contradicts Last Take Wins → Last Take Wins unless score gap > 20%
  - If removal would leave < 3 words in a segment → don't remove (fragment protection)

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
SIMILARITY_THRESHOLD = 0.70
LOOKBACK_SECONDS = 15.0
SCORING_OVERRIDE_MARGIN = 0.20
TARGET_WPS = 2.7
COARTICULATION_GAP_MS = 20.0
MIN_SEGMENT_WORDS = 3


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
    Detect any word repeated 2+ times consecutively with gap < 300ms.
    Keep only the last occurrence.
    """
    decisions = []
    remove_ids = set()
    i = 0

    while i < len(presenter_words):
        w = presenter_words[i]
        text = w["word"].strip()

        # Find consecutive identical words with small gap
        group = [w]
        j = i + 1
        while j < len(presenter_words):
            nw = presenter_words[j]
            gap_ms = (nw["start"] - group[-1]["end"]) * 1000.0
            if nw["word"].strip() == text and gap_ms < 300:
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

    # Split into sentences by gaps > 1500ms
    sentences = []
    current = []
    for i, w in enumerate(presenter_words):
        if w["id"] in already_removed:
            continue
        if current:
            gap_ms = (w["start"] - current[-1]["end"]) * 1000.0
            if gap_ms > 1500:
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

        if len(sent) >= 6:
            continue

        # Check no punctuation at end
        last_word = sent[-1]["word"].strip()
        if ends_with_punctuation(last_word):
            continue

        # Check gap after
        gap_ms = (next_sent[0]["start"] - sent[-1]["end"]) * 1000.0
        if gap_ms < 1500:
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


def build_takes(words, already_removed, gap_threshold=0.5):
    """
    Group active words into complete takes based on silence gaps.
    A gap > gap_threshold (seconds) between consecutive words = take boundary.
    Returns list of takes, each take is a list of word dicts.
    """
    active = [w for w in words if w["id"] not in already_removed]
    if not active:
        return []

    takes = []
    current_take = [active[0]]

    for i in range(1, len(active)):
        gap = active[i]["start"] - current_take[-1]["end"]
        if gap > gap_threshold:
            takes.append(current_take)
            current_take = [active[i]]
        else:
            current_take.append(active[i])

    if current_take:
        takes.append(current_take)

    return takes


def find_similar_take_groups(takes, threshold, lookback_sec):
    """
    Compare all take pairs within lookback window using cosine BoW.
    Group similar takes (similarity >= threshold) using greedy chaining.
    Returns list of groups, each group is a list of take indices.
    """
    n = len(takes)
    # Build adjacency: which takes are similar to each other
    adjacency = {i: set() for i in range(n)}

    for i in range(n):
        text_i = [w["word"].strip() for w in takes[i]]
        start_i = takes[i][0]["start"]
        end_i = takes[i][-1]["end"]

        for j in range(i + 1, n):
            start_j = takes[j][0]["start"]
            # Check lookback window
            if start_j - end_i > lookback_sec:
                break

            text_j = [w["word"].strip() for w in takes[j]]
            sim = cosine_similarity_bow(text_i, text_j)
            if sim >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Greedy connected-component grouping
    visited = set()
    groups = []

    for i in range(n):
        if i in visited or not adjacency[i]:
            continue
        # BFS to find connected component
        group = []
        queue = [i]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(group) >= 2:
            groups.append(sorted(group))

    return groups


def detect_repetitions(presenter_words, already_removed, threshold, lookback_sec):
    """
    Section 4.1: Complete-take repetition detection.
    Groups words into takes (by silence > 500ms), compares whole takes
    using cosine similarity, and removes all but the last take in each
    similar group. Never removes individual words from within a take.
    """
    decisions = []
    remove_ids = set()

    # Step 1: Build takes from active words
    takes = build_takes(presenter_words, already_removed)
    log(f"Repetition detection: {len(takes)} takes identified")

    if len(takes) < 2:
        return remove_ids, decisions

    # Step 2: Find groups of similar takes
    groups = find_similar_take_groups(takes, threshold, lookback_sec)
    log(f"Repetition detection: {len(groups)} similar-take groups found")

    # Step 3: For each group, keep last take, remove the rest
    for group in groups:
        # Sort by start time — last take wins
        group_takes = [(idx, takes[idx]) for idx in group]
        group_takes.sort(key=lambda t: t[1][0]["start"])

        kept_take = group_takes[-1][1]
        kept_ids = sorted([w["id"] for w in kept_take])

        for idx, take in group_takes[:-1]:
            ids = sorted([w["id"] for w in take])
            remove_ids.update(ids)
            decisions.append({
                "ids": ids,
                "reason": "duplicate_take",
                "kept_ids": kept_ids,
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


# ── Priority & Protection ────────────────────────────

def detect_internal_retakes(words, already_removed):
    """
    Scan within each take for repeated 3+ word sequences.
    If "פרויקט חד פעמי" appears at positions 5-7 AND again at 15-17
    within the same take — remove the FIRST occurrence (keep last).
    """
    remove_ids = set()
    decisions = []

    active = [w for w in words if w['id'] not in already_removed]

    # Build takes (gap > 500ms = new take)
    takes = build_takes(active, already_removed)

    for take in takes:
        if len(take) < 6:  # too short for internal retakes
            continue

        take_words = [w['word'] for w in take]

        # Sliding window: look for 3-gram that appears twice
        for i in range(len(take_words) - 2):
            trigram = take_words[i:i+3]
            # Search for same trigram later in the take
            for j in range(i + 3, len(take_words) - 2):
                if take_words[j:j+3] == trigram:
                    # Found internal retake! Remove from i to j-1
                    ids_to_remove = [take[k]['id'] for k in range(i, j)]
                    remove_ids.update(ids_to_remove)
                    decisions.append({
                        'ids': sorted(ids_to_remove),
                        'reason': 'internal_retake'
                    })
                    break  # only handle first match per trigram
            if remove_ids:
                break  # restart scan after finding one

    return remove_ids, decisions


def apply_hard_rejection(words, all_words, already_removed):
    """
    Hard reject entire segments that fail quality checks (2+ signals required).
    These segments will be marked for removal and NOT sent to AI.
    """
    reject_ids = set()
    decisions = []

    active = [w for w in words if w['id'] not in already_removed]
    takes = build_takes(active, already_removed)

    for take in takes:
        if not take:
            continue
        take_ids = [w['id'] for w in take]

        # Count how many rejection signals this take has
        signals = 0
        signal_names = []

        # SIGNAL 1: >40% of words are non-presenter
        non_presenter = sum(1 for w in take if not w.get('is_presenter', True))
        if len(take) > 0 and non_presenter / len(take) > 0.4:
            signals += 1
            signal_names.append('non_presenter')

        # SIGNAL 2: speaker_score average < 0.45
        scores = [w.get('speaker_score', 1.0) for w in take if 'speaker_score' in w]
        if scores and sum(scores) / len(scores) < 0.45:
            signals += 1
            signal_names.append('low_speaker_score')

        # SIGNAL 3: Incomplete ending (last word <= 2 chars + gap > 1s)
        last_word = take[-1]
        incomplete_ending = False
        if len(last_word['word']) <= 2:
            next_words = [w for w in all_words if w['start'] > last_word['end']]
            if next_words and next_words[0]['start'] - last_word['end'] > 1.0:
                incomplete_ending = True
        if incomplete_ending:
            signals += 1
            signal_names.append('incomplete_ending')

        # SIGNAL 4: Heavy stuttering (same word repeated consecutively)
        word_texts = [w['word'] for w in take]
        stutter_count = sum(1 for i in range(len(word_texts) - 1)
                           if word_texts[i] == word_texts[i + 1])
        if len(take) > 3 and stutter_count / len(take) > 0.2:
            signals += 1
            signal_names.append('heavy_stutter')

        # Reject only if 2+ signals
        if signals >= 2:
            reject_ids.update(take_ids)
            decisions.append({
                'ids': sorted(take_ids),
                'reason': 'hard_reject',
                'signals': signal_names,
            })
            log(f"Hard reject: {len(take_ids)} words, signals: {signal_names}")

    return reject_ids, decisions


def build_ranked_candidates(words, all_words, already_removed, audio_data=None, sample_rate=16000):
    """
    For each unique sentence/idea, find all takes and rank them.
    Returns top 3 candidates per sentence group.
    """
    active = [w for w in words if w['id'] not in already_removed
              and w.get('is_presenter', True)]
    takes = build_takes(active, already_removed)

    if not takes:
        return []

    # Group similar takes
    groups = find_similar_take_groups(takes, threshold=0.70, lookback_sec=999)

    ranked = []
    for group_indices in groups:
        group_takes = [takes[i] for i in group_indices]
        scored = []
        for take in group_takes:
            score = score_take(take, audio_data, sample_rate)

            # Bonus: no starred words
            starred = sum(1 for w in take if not w.get('is_presenter', True))
            clean_bonus = 0.1 if starred == 0 else 0

            # Bonus: complete sentence (last word > 2 chars)
            last_word = take[-1]['word']
            complete_bonus = 0.05 if len(last_word) > 2 else 0

            total = score + clean_bonus + complete_bonus
            scored.append((take, total))

        # Sort by score descending, keep top 3
        scored.sort(key=lambda x: x[1], reverse=True)

        best_take = scored[0][0]
        sentence_text = ' '.join(w['word'] for w in best_take[:6])
        if len(best_take) > 6:
            sentence_text += '...'

        entry = {
            'sentence_group': sentence_text,
            'best_take_ids': [w['id'] for w in scored[0][0]],
            'alternatives': [
                {
                    'ids': [w['id'] for w in take],
                    'score': round(s, 2),
                }
                for take, s in scored[1:3]
            ],
        }
        ranked.append(entry)

    log(f"Ranked candidates: {len(ranked)} sentence groups")
    return ranked


def apply_coarticulation_protection(remove_ids, presenter_words, decisions):
    """
    CRITICAL rule 3: If gap between two consecutive words < 20ms,
    they are coarticulated — NEVER cut between them.
    If one is marked for removal and the other isn't, restore the marked one.

    Coarticulation protection applies ONLY to individual word removals
    (stutters, isolated fillers). NOT to entire duplicate takes,
    production cues, or abandoned takes. If a word is part of a group
    removal → coarticulation does NOT restore it.
    """
    # Build set of word IDs that belong to group removals (not individual)
    group_removal_reasons = {"duplicate_take", "production_cue", "abandoned_take", "false_start"}
    group_removal_ids = set()
    for dec in decisions:
        if dec["reason"] in group_removal_reasons:
            group_removal_ids.update(dec["ids"])

    restored = set()
    for i in range(len(presenter_words) - 1):
        curr = presenter_words[i]
        nxt = presenter_words[i + 1]
        gap_ms = (nxt["start"] - curr["end"]) * 1000.0

        if gap_ms < COARTICULATION_GAP_MS:
            curr_removed = curr["id"] in remove_ids
            nxt_removed = nxt["id"] in remove_ids
            # If only one of the pair is removed → restore it (don't cut)
            # But skip if the word is part of a group removal
            if curr_removed and not nxt_removed and curr["id"] not in group_removal_ids:
                restored.add(curr["id"])
            elif nxt_removed and not curr_removed and nxt["id"] not in group_removal_ids:
                restored.add(nxt["id"])

    if restored:
        remove_ids -= restored
        log(f"Coarticulation protection: restored {len(restored)} words (gap < {COARTICULATION_GAP_MS}ms)")

    return remove_ids, restored


def apply_fragment_protection(remove_ids, presenter_words, decisions):
    """
    If removal would leave < MIN_SEGMENT_WORDS in a segment, don't remove.
    Segments are groups of consecutive kept words (gap < 600ms).

    Fragment protection applies ONLY when removing isolated words,
    NOT when removing entire duplicate takes, production cues, or
    abandoned takes.
    """
    # Build set of word IDs that belong to group removals
    group_removal_reasons = {"duplicate_take", "production_cue", "abandoned_take", "false_start"}
    group_removal_ids = set()
    for dec in decisions:
        if dec["reason"] in group_removal_reasons:
            group_removal_ids.update(dec["ids"])

    # Build segments of remaining words
    remaining = [w for w in presenter_words if w["id"] not in remove_ids]
    if not remaining:
        return remove_ids

    segments = []
    current_seg = [remaining[0]]
    for i in range(1, len(remaining)):
        gap_ms = (remaining[i]["start"] - current_seg[-1]["end"]) * 1000.0
        if gap_ms > 600:
            segments.append(current_seg)
            current_seg = [remaining[i]]
        else:
            current_seg.append(remaining[i])
    segments.append(current_seg)

    # Check each segment
    restored = set()
    for seg in segments:
        if len(seg) < MIN_SEGMENT_WORDS:
            seg_start = seg[0]["start"]
            seg_end = seg[-1]["end"]
            for w in presenter_words:
                if w["id"] not in remove_ids:
                    continue
                # Skip words that are part of group removals
                if w["id"] in group_removal_ids:
                    continue
                if (seg_start - 0.6) <= w["start"] <= (seg_end + 0.6):
                    restored.add(w["id"])

    if restored:
        remove_ids -= restored
        log(f"Fragment protection: restored {len(restored)} words (segments < {MIN_SEGMENT_WORDS} words)")

    return remove_ids


def deduplicate_decisions(decisions, remove_ids):
    """
    Clean up decisions: remove IDs that were restored by protections,
    and ensure production_cue reason wins over others for shared IDs.
    """
    # Build priority map: word_id → highest priority reason
    # Priority: production_cue > abandoned_take > duplicate_take > false_start > stutter
    priority = {
        "production_cue": 5,
        "abandoned_take": 4,
        "duplicate_take": 3,
        "false_start": 2,
        "stutter": 1,
    }

    word_reason = {}
    for dec in decisions:
        for wid in dec["ids"]:
            if wid not in remove_ids:
                continue
            existing_priority = priority.get(word_reason.get(wid, ""), 0)
            new_priority = priority.get(dec["reason"], 0)
            if new_priority > existing_priority:
                word_reason[wid] = dec["reason"]

    # Rebuild decisions grouped by reason
    reason_groups = {}
    for wid, reason in word_reason.items():
        if reason not in reason_groups:
            reason_groups[reason] = []
        reason_groups[reason].append(wid)

    # Find kept_ids from original decisions
    kept_map = {}
    for dec in decisions:
        if "kept_ids" in dec:
            for wid in dec["ids"]:
                if wid in remove_ids:
                    kept_map[wid] = dec["kept_ids"]

    cleaned = []
    for reason, ids in reason_groups.items():
        ids.sort()
        entry = {"ids": ids, "reason": reason}
        # Attach kept_ids from first matching decision
        for wid in ids:
            if wid in kept_map:
                entry["kept_ids"] = kept_map[wid]
                break
        cleaned.append(entry)

    return cleaned


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

    # Use ALL words for pattern detection (n-gram continuity),
    # but only mark presenter words for removal
    all_words = sorted(words, key=lambda w: w["start"])
    presenter_ids = {w["id"] for w in all_words if w.get("is_presenter", True)}

    log(f"Analyzing {len(all_words)} total words ({len(presenter_ids)} presenter)...")

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
    stutter_ids, stutter_decs = detect_micro_stutters(all_words)
    all_remove_ids |= stutter_ids
    all_decisions.extend(stutter_decs)
    log(f"Micro-stutters: {len(stutter_decs)} groups, {len(stutter_ids)} words")

    # 2. Production cues (4.4)
    cue_ids, cue_decs = detect_production_cues(all_words)
    all_remove_ids |= cue_ids
    all_decisions.extend(cue_decs)
    log(f"Production cues: {len(cue_decs)} found, {len(cue_ids)} words")

    # 3. Full abandoned takes (4.5)
    abandoned_ids, abandoned_decs = detect_full_abandoned_takes(all_words, all_remove_ids)
    all_remove_ids |= abandoned_ids
    all_decisions.extend(abandoned_decs)
    log(f"Abandoned takes: {len(abandoned_decs)} found, {len(abandoned_ids)} words")

    # 4. False starts (4.2)
    false_start_ids, false_start_decs = detect_false_starts(all_words, all_remove_ids)
    all_remove_ids |= false_start_ids
    all_decisions.extend(false_start_decs)
    log(f"False starts: {len(false_start_decs)} found, {len(false_start_ids)} words")

    # 5. Repetition detection (4.1)
    rep_ids, rep_decs = detect_repetitions(
        all_words, all_remove_ids,
        args.similarity_threshold, args.lookback_seconds,
    )
    all_remove_ids |= rep_ids
    all_decisions.extend(rep_decs)
    log(f"Repetitions: {len(rep_decs)} groups, {len(rep_ids)} words")

    # 6. Take scoring (4.6) — refine duplicate decisions
    all_decisions = apply_take_scoring(
        all_decisions, all_words,
        audio_data, sample_rate,
        args.scoring_override_margin,
    )

    # Recompute remove_ids after scoring may have swapped takes
    all_remove_ids = set()
    for dec in all_decisions:
        all_remove_ids.update(dec["ids"])

    # 6.5. Hard rejection (requires 2+ signals to reject)
    hard_reject_ids, hard_reject_decs = apply_hard_rejection(all_words, all_words, all_remove_ids)
    all_remove_ids |= hard_reject_ids
    all_decisions.extend(hard_reject_decs)
    log(f"Hard rejections: {len(hard_reject_decs)} takes, {len(hard_reject_ids)} words")

    # 6.7. Internal retake detection (within-take repeated phrases)
    internal_ids, internal_decs = detect_internal_retakes(all_words, all_remove_ids)
    all_remove_ids |= internal_ids
    all_decisions.extend(internal_decs)
    log(f"Internal retakes: {len(internal_decs)} found, {len(internal_ids)} words")

    # Filter: only remove presenter words (non-presenter already handled by merge)
    non_presenter_filtered = all_remove_ids - presenter_ids
    if non_presenter_filtered:
        log(f"Filtered out {len(non_presenter_filtered)} non-presenter words from removal")
        all_remove_ids &= presenter_ids

    pre_protection_groups = len(all_decisions)
    pre_protection_words = len(all_remove_ids)
    log(f"Before protections: {pre_protection_groups} groups ({pre_protection_words} words)")

    # 7. Coarticulation protection (CRITICAL rule 3)
    #    Only applies to individual word removals (stutters), NOT group removals
    all_remove_ids, coart_restored = apply_coarticulation_protection(all_remove_ids, all_words, all_decisions)

    # 8. Fragment protection
    #    Only applies to individual word removals, NOT group removals
    all_remove_ids = apply_fragment_protection(all_remove_ids, all_words, all_decisions)

    post_protection_words = len(all_remove_ids)
    restored_count = pre_protection_words - post_protection_words
    log(f"After protections: {pre_protection_groups} groups ({post_protection_words} words) — {restored_count} restored")

    # Ensure only presenter words remain in remove_ids after protections
    all_remove_ids &= presenter_ids

    # 8.5. Build ranked candidates (for AI referee)
    candidates = build_ranked_candidates(all_words, all_words, all_remove_ids, audio_data, sample_rate or 16000)

    # 9. Deduplicate and clean decisions (priority-based conflict resolution)
    all_decisions = deduplicate_decisions(all_decisions, all_remove_ids)

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
        "internal_retakes": reason_counts.get("internal_retake", 0),
        "hard_rejections": reason_counts.get("hard_reject", 0),
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
        "candidates": candidates,
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

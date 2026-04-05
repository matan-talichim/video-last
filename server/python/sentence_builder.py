#!/usr/bin/env python3
"""
Sentence Builder — Builds a menu of complete, scored, deduplicated sentences
from merged transcript + take decisions.

The AI then selects from this menu by sentence ID instead of picking word IDs.

CLI:
    python3 sentence_builder.py \
        --merged merged_transcript.json \
        --take-decisions take_decisions.json \
        --output sentences.json

Output: JSON with sentences array and stats.
Logs to stderr.
"""

import argparse
import json
import sys
import time


# ── Constants ────────────────────────────────────────

# Hebrew words that typically start a new sentence
SENTENCE_OPENERS = {
    "אם", "אנחנו", "אני", "אתה", "אתם", "זה", "זאת",
    "הם", "היא", "הוא", "כל", "כש", "למה", "מה", "מי",
    "ואם", "ואנחנו", "כשאתה", "כשאני", "מאחורי",
    "העסק", "והעסק", "תשאיר", "יותר", "פרויקט",
    "בוא", "בואו", "היום", "כדי", "לכן", "עכשיו",
    "ברגע", "תחשוב", "תחשבו", "אז", "הבעיה", "הפתרון",
    "במקום", "בגלל", "השאלה", "התשובה", "המטרה",
}

# Words that indicate an incomplete sentence ending
INCOMPLETE_ENDINGS = {
    "את", "של", "על", "עם", "לא", "הוא", "היא", "זה",
    "ולא", "אני", "שלך", "שלו", "שלה", "שלנו", "בלי",
    "כמו", "אבל", "כי", "ש", "וגם", "או", "אם",
}

# Minimum words per sentence
MIN_SENTENCE_WORDS = 4

# Gap threshold (seconds) for sentence splitting within a take
SENTENCE_GAP_THRESHOLD = 0.15

# Dedup similarity threshold — only near-identical sentences are removed
DEDUP_SIMILARITY_THRESHOLD = 0.85

# Words that indicate a fragment (not a complete sentence) when they appear first
FRAGMENT_STARTS = {
    "לא", "את", "של", "על", "עם", "הוא", "היא", "ולא",
    "שהחיסכון", "לך", "שלך", "שלו", "שלה", "שלנו",
    "גם", "וגם", "או", "כי", "ש", "אבל",
}


def log(msg):
    """Log to stderr."""
    print(f"[sentence_builder] {msg}", file=sys.stderr)


# ── Step 1: Filter ────────────────────────────────────

def filter_words(words, remove_ids):
    """
    Remove words that are:
    - in take_decisions.remove_ids (stutters, cues, duplicates)
    - final_decision == 'reject' (non-presenter)
    Keep: presenter + uncertain words.
    """
    clean = [
        w for w in words
        if w["id"] not in remove_ids
        and w.get("final_decision") != "reject"
    ]
    log(f"Filter: {len(words)} → {len(clean)} words "
        f"(removed {len(words) - len(clean)})")
    return clean


# ── Step 2: Group by take ─────────────────────────────

def group_by_take(words):
    """Group consecutive words by take_id."""
    takes = {}
    for word in words:
        tid = word.get("take_id", 0)
        if tid not in takes:
            takes[tid] = []
        takes[tid].append(word)

    log(f"Group by take: {len(takes)} takes")
    return takes


# ── Step 3: Split takes into sentences ────────────────

def split_take_to_sentences(take_words):
    """
    Split a take's words into sentences using:
    - Gaps > SENTENCE_GAP_THRESHOLD between words
    - Hebrew sentence openers
    """
    if not take_words:
        return []

    sentences = []
    current = []

    for i, word in enumerate(take_words):
        # Start new sentence if:
        # 1. Current has enough words AND
        # 2. This word is a sentence opener AND
        # 3. Gap from previous word > threshold
        if current and len(current) >= MIN_SENTENCE_WORDS:
            gap = (word["start"] - take_words[i - 1]["end"]) if i > 0 else 0
            word_text = word["word"].strip()
            if word_text in SENTENCE_OPENERS and gap > SENTENCE_GAP_THRESHOLD:
                sentences.append(current)
                current = []

        current.append(word)

    if current and len(current) >= MIN_SENTENCE_WORDS:
        sentences.append(current)

    return sentences


def filter_fragments(sentence_word_lists):
    """
    Remove fragments — sentences whose first word is a function word
    that indicates an incomplete/broken sentence.
    """
    kept = []
    removed = 0
    for sent_words in sentence_word_lists:
        first_word = sent_words[0]["word"].strip()
        if first_word in FRAGMENT_STARTS:
            text = " ".join(w["word"] for w in sent_words)
            log(f"Fragment filtered: \"{text}\" (starts with '{first_word}')")
            removed += 1
        else:
            kept.append(sent_words)
    if removed:
        log(f"Fragment filter: removed {removed}, kept {len(kept)}")
    return kept


# ── Step 4: Score each sentence ───────────────────────

def score_sentence(sentence_words):
    """
    Score based on:
    - avg speaker_score (35%)
    - completeness: does it end with a 'closing' word? (25%)
    - fluency: no big gaps within sentence (20%)
    - word count: prefer 5-15 words (10%)
    - take number: higher = later = usually better (10%)
    """
    if not sentence_words:
        return 0.0

    # Speaker score average
    scores = [w.get("speaker_score", 0.5) for w in sentence_words]
    avg_spk = sum(scores) / len(scores)

    # Completeness — last word should not be a function word
    last_word = sentence_words[-1]["word"].strip()
    completeness = 0.0 if last_word in INCOMPLETE_ENDINGS else 1.0

    # Fluency — no big gaps within sentence
    max_gap = 0.0
    for i in range(1, len(sentence_words)):
        gap = sentence_words[i]["start"] - sentence_words[i - 1]["end"]
        max_gap = max(max_gap, gap)
    if max_gap < 0.5:
        fluency = 1.0
    elif max_gap < 1.0:
        fluency = 0.5
    else:
        fluency = 0.0

    # Word count — prefer 5-15
    wc = len(sentence_words)
    if 5 <= wc <= 15:
        word_score = 1.0
    elif 3 <= wc <= 20:
        word_score = 0.7
    else:
        word_score = 0.3

    # Take number — higher is better
    take_id = sentence_words[0].get("take_id", 0)
    take_score = min(take_id / 20.0, 1.0)

    return (
        avg_spk * 0.35
        + completeness * 0.25
        + fluency * 0.20
        + word_score * 0.10
        + take_score * 0.10
    )


# ── Step 5: Deduplicate ──────────────────────────────

def deduplicate_sentences(sentences):
    """
    Compare all sentences using semantic similarity.
    If two sentences are > DEDUP_SIMILARITY_THRESHOLD similar — keep the one with higher score.
    """
    from take_selector import semantic_similarity

    # Sort by score descending — higher scores kept first
    sorted_sents = sorted(sentences, key=lambda s: s["score"], reverse=True)

    unique = []
    for sent in sorted_sents:
        is_dup = False
        sent_words = sent["text"].split()
        for kept in unique:
            kept_words = kept["text"].split()
            sim = semantic_similarity(sent_words, kept_words)
            if sim > DEDUP_SIMILARITY_THRESHOLD:
                is_dup = True
                log(f"Dedup: removing sentence (score {sent['score']:.2f}) "
                    f"similar to kept (score {kept['score']:.2f}): "
                    f"\"{sent['text'][:40]}...\" ~ \"{kept['text'][:40]}...\"")
                break
        if not is_dup:
            unique.append(sent)

    log(f"Dedup: {len(sentences)} → {len(unique)} sentences")
    return unique


# ── Main: build_sentences ─────────────────────────────

def build_sentences(words, take_decisions):
    """
    Main orchestrator:
    1. Filter words (remove stutters, cues, rejected)
    2. Group by take
    3. Split into sentences
    4. Score each sentence
    5. Deduplicate
    6. Return numbered menu
    """
    t0 = time.time()

    # Step 1: Filter
    remove_ids = set(take_decisions.get("remove_ids", []))
    clean_words = filter_words(words, remove_ids)

    if not clean_words:
        log("WARNING: No words left after filtering!")
        return {"sentences": [], "stats": {
            "total_takes": 0, "total_raw_sentences": 0,
            "after_dedup": 0, "total_duration": 0, "avg_score": 0,
            "processing_time_ms": int((time.time() - t0) * 1000),
        }}

    # Step 2: Group by take
    takes = group_by_take(clean_words)

    # Step 3: Split into sentences
    all_sentences = []
    for tid in sorted(takes.keys()):
        take_words = takes[tid]
        sents = split_take_to_sentences(take_words)
        # Step 3b: Filter fragments
        sents = filter_fragments(sents)
        for sent_words in sents:
            text = " ".join(w["word"] for w in sent_words)
            word_ids = [w["id"] for w in sent_words]
            take_id = sent_words[0].get("take_id", tid)
            start = sent_words[0]["start"]
            end = sent_words[-1]["end"]
            duration = end - start
            spk_scores = [w.get("speaker_score", 0.5) for w in sent_words]
            avg_spk = sum(spk_scores) / len(spk_scores)

            # Score
            score = score_sentence(sent_words)

            # Completeness for output
            last_word = sent_words[-1]["word"].strip()
            completeness = 0.0 if last_word in INCOMPLETE_ENDINGS else 1.0

            all_sentences.append({
                "text": text,
                "word_ids": word_ids,
                "take_id": take_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "score": round(score, 3),
                "word_count": len(sent_words),
                "speaker_score_avg": round(avg_spk, 3),
                "completeness": completeness,
            })

    total_raw = len(all_sentences)
    log(f"Raw sentences: {total_raw} from {len(takes)} takes")
    for idx, s in enumerate(all_sentences):
        log(f"  raw[{idx}] (score {s['score']:.2f}, {s['word_count']}w): \"{s['text']}\"")

    # Step 5: Deduplicate
    unique_sentences = deduplicate_sentences(all_sentences)

    # Sort by timeline (start time)
    unique_sentences.sort(key=lambda s: s["start"])

    # Assign sequential IDs
    for i, sent in enumerate(unique_sentences):
        sent["id"] = i + 1

    # Stats
    total_duration = sum(s["duration"] for s in unique_sentences)
    avg_score = (
        sum(s["score"] for s in unique_sentences) / len(unique_sentences)
        if unique_sentences else 0
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    log(f"Done: {len(unique_sentences)} sentences, "
        f"total duration {total_duration:.1f}s, "
        f"avg score {avg_score:.2f}, "
        f"took {elapsed_ms}ms")

    return {
        "sentences": unique_sentences,
        "stats": {
            "total_takes": len(takes),
            "total_raw_sentences": total_raw,
            "after_dedup": len(unique_sentences),
            "total_duration": round(total_duration, 1),
            "avg_score": round(avg_score, 3),
            "processing_time_ms": elapsed_ms,
        },
    }


# ── CLI ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build sentence menu from merged transcript + take decisions"
    )
    parser.add_argument("--merged", required=True, help="Path to merged_transcript.json")
    parser.add_argument("--take-decisions", required=True, help="Path to take_decisions.json")
    parser.add_argument("--output", required=True, help="Output path for sentences.json")
    args = parser.parse_args()

    # Load inputs
    with open(args.merged, "r", encoding="utf-8") as f:
        merged = json.load(f)

    with open(args.take_decisions, "r", encoding="utf-8") as f:
        take_decisions = json.load(f)

    words = merged.get("words", [])
    log(f"Loaded {len(words)} words from merged transcript")

    # Build sentences
    result = build_sentences(words, take_decisions)

    # Write output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print JSON to stdout for TypeScript to read
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

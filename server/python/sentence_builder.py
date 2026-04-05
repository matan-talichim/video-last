#!/usr/bin/env python3
"""
Sentence Builder — converts word-level transcript into a sentence menu for AI selection.

Instead of giving the AI a raw numbered word list and hoping it builds coherent segments,
this module pre-groups words into complete sentences, scores them, deduplicates similar
versions, and outputs a clean "sentence menu" that the AI can simply pick from.

Pipeline position: AFTER take_selector, BEFORE AI narrative selection.

Usage:
  python3 sentence_builder.py \
    --merged input/$JOB/merged_transcript.json \
    --take-decisions input/$JOB/take_decisions.json \
    --output /tmp/sentence_menu.json

Output: sentence_menu.json with structure:
  {
    "sentences": [
      {
        "id": "S1",
        "text": "אם כל לקוח חדש אצלך גורם לך לחשוב...",
        "word_ids": [12, 13, 14, ...],
        "take_id": 3,
        "start": 5.2,
        "end": 8.1,
        "word_count": 12,
        "score": 0.92,
        "scores": { "speaker": 0.88, "asr": 0.95, "completeness": 1.0, "energy": 0.85 },
        "group_id": "G1",
        "is_best_in_group": true,
        "alternatives": 2,
        "tags": ["HOOK"]
      }
    ],
    "groups": [
      {
        "group_id": "G1",
        "representative_text": "אם כל לקוח חדש...",
        "version_count": 3,
        "best_sentence_id": "S1",
        "all_sentence_ids": ["S1", "S5", "S9"]
      }
    ],
    "stats": { ... }
  }
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict


# ── Constants ──────────────────────────────────────

# Gap threshold to split sentences within a take (seconds)
SENTENCE_GAP_THRESHOLD = 1.5

# Minimum words for a valid sentence
MIN_SENTENCE_WORDS = 3

# Similarity threshold for grouping sentences as "versions" of the same content
SIMILARITY_THRESHOLD = 0.70

# Production cues — sentences containing only these are filtered out
PRODUCTION_CUES = [
    "נתחיל שוב", "עוד פעם", "רגע", "סליחה", "מהתחלה",
    "יאללה", "מוכנים", "סיימתי", "יופי", "מעולה",
    "עוד טייק", "טייק", "שנייה", "אחד שתיים שלוש",
    "בדיקה", "ok", "okay", "ready",
]

# Hebrew filler words (standalone only)
HEBREW_FILLERS = {
    "אה", "אהה", "אמ", "אממ", "כאילו", "נו",
    "אוקיי", "סתם", "יאללה", "ככה", "כזה",
}

# Hebrew punctuation that marks sentence boundaries
SENTENCE_TERMINATORS = {".", "?", "!", "׃"}

# Words that should NEVER start a sentence (mid-sentence connectors)
FORBIDDEN_STARTS = {
    "זמן", "וכסף", "ידנית", "שלך", "לך", "שהחיסכון",
    "פרטים", "בחינם", "מחליף", "והופכים", "אותם",
    "ולא", "ובלי", "שגונבים", "ומחכה", "ואם", "שיהיה",
}


# ── Helpers ────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Build sentence menu from merged transcript")
    parser.add_argument("--merged", required=True, help="Path to merged_transcript.json")
    parser.add_argument("--take-decisions", required=True, help="Path to take_decisions.json")
    parser.add_argument("--output", required=True, help="Path to output sentence_menu.json")
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Cosine similarity threshold for grouping (default: 0.70)")
    parser.add_argument("--min-words", type=int, default=MIN_SENTENCE_WORDS,
                        help="Minimum words per sentence (default: 3)")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_production_cue(text: str) -> bool:
    """Check if entire sentence text is a production cue."""
    normalized = text.strip().lower()
    for cue in PRODUCTION_CUES:
        if normalized == cue.lower() or normalized.startswith(cue.lower()):
            return True
    return False


def is_filler_only(words: list) -> bool:
    """Check if all words in the sentence are fillers."""
    return all(w["word"].strip() in HEBREW_FILLERS for w in words)


def has_forbidden_start(words: list) -> bool:
    """Check if sentence starts with a forbidden connector word."""
    if not words:
        return False
    first_word = words[0]["word"].strip()
    return first_word in FORBIDDEN_STARTS


# ── Sentence Splitting ─────────────────────────────

def split_take_to_sentences(words: list, gap_threshold: float = SENTENCE_GAP_THRESHOLD) -> list:
    """
    Split a sequence of words (all from one take) into sentences.

    Boundaries are determined by:
    1. Punctuation at word end (., ?, !, ׃)
    2. Time gap > gap_threshold between consecutive words
    3. Hebrew comma + gap > 0.8s (natural clause boundary)
    """
    if not words:
        return []

    sentences = []
    current = []

    for i, word in enumerate(words):
        current.append(word)

        is_last = (i == len(words) - 1)
        should_break = False

        if not is_last:
            next_word = words[i + 1]
            gap = next_word["start"] - word["end"]
            word_text = word["word"].strip()

            # Rule 1: Punctuation terminator
            if any(word_text.endswith(t) for t in SENTENCE_TERMINATORS):
                should_break = True

            # Rule 2: Large time gap
            elif gap > gap_threshold:
                should_break = True

            # Rule 3: Comma + moderate gap (clause boundary)
            elif word_text.endswith(",") and gap > 0.8:
                should_break = True

        if should_break or is_last:
            if len(current) >= 1:  # We'll filter by min_words later
                sentences.append(current)
            current = []

    return sentences


# ── Scoring ────────────────────────────────────────

def score_sentence(words: list) -> dict:
    """
    Compute a composite score for a sentence.

    Components:
    - speaker_score: average speaker_score (voice identity match)
    - asr_score: average ASR confidence
    - energy_score: average RMS energy
    - completeness: 1.0 if >= 4 words, degrades below
    - gap_penalty: penalty for internal gaps > 800ms
    """
    if not words:
        return {"speaker": 0, "asr": 0, "energy": 0, "completeness": 0, "gap_penalty": 0, "total": 0}

    # Average scores
    speaker_scores = [w.get("speaker_score", 0.5) for w in words]
    asr_scores = [w.get("confidence", 0.5) for w in words]
    energy_scores = [w.get("energy_score", 0.5) for w in words]

    avg_speaker = sum(speaker_scores) / len(speaker_scores) if speaker_scores else 0.5
    avg_asr = sum(asr_scores) / len(asr_scores) if asr_scores else 0.5
    avg_energy = sum(energy_scores) / len(energy_scores) if energy_scores else 0.5

    # Completeness: sentences >= 4 words get full score
    word_count = len(words)
    completeness = min(1.0, word_count / 4.0)

    # Gap penalty: check for internal silence > 800ms
    gap_penalty = 0.0
    for i in range(1, len(words)):
        internal_gap = words[i]["start"] - words[i - 1]["end"]
        if internal_gap > 0.8:
            gap_penalty += 0.05 * (internal_gap - 0.8)

    gap_penalty = min(gap_penalty, 0.3)  # Cap at 0.3

    # Weighted composite
    total = (
        avg_speaker * 0.35 +
        avg_asr * 0.25 +
        avg_energy * 0.15 +
        completeness * 0.25 -
        gap_penalty
    )

    return {
        "speaker": round(avg_speaker, 3),
        "asr": round(avg_asr, 3),
        "energy": round(avg_energy, 3),
        "completeness": round(completeness, 3),
        "gap_penalty": round(gap_penalty, 3),
        "total": round(max(0, min(1.0, total)), 3),
    }


# ── Similarity / Deduplication ─────────────────────

def tokenize_hebrew(text: str) -> list:
    """Simple tokenizer: split on whitespace, remove punctuation, lowercase."""
    tokens = text.strip().split()
    cleaned = []
    for t in tokens:
        t = re.sub(r'[^\w\s]', '', t).strip()
        if t:
            cleaned.append(t)
    return cleaned


def compute_tf(tokens: list) -> dict:
    """Compute term frequency vector."""
    tf = defaultdict(float)
    for t in tokens:
        tf[t] += 1.0
    # Normalize
    total = len(tokens)
    if total > 0:
        for t in tf:
            tf[t] /= total
    return dict(tf)


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse TF vectors."""
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    if not all_keys:
        return 0.0

    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def group_similar_sentences(sentences: list, threshold: float) -> list:
    """
    Group sentences by text similarity (cosine on TF vectors).
    Returns list of groups, each group is a list of sentence indices.
    """
    n = len(sentences)
    if n == 0:
        return []

    # Pre-compute TF vectors
    tf_vectors = []
    for s in sentences:
        tokens = tokenize_hebrew(s["text"])
        tf_vectors.append(compute_tf(tokens))

    # Union-Find for grouping
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(tf_vectors[i], tf_vectors[j])
            if sim >= threshold:
                union(i, j)

    # Build groups
    groups_map = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    return list(groups_map.values())


# ── Tag Assignment ─────────────────────────────────

# Pattern-based tagging for narrative structure
HOOK_PATTERNS = [
    r"^אם\s",           # "אם כל לקוח..." — conditional hook
    r"^מה\s(אם|היה)",    # "מה אם..." — what-if hook
    r"^תארו?\s",         # "תארו לעצמכם" — imagine hook
    r"^האם\s",           # "האם אתה..." — question hook
    r"\?$",              # Ends with question mark
]

CTA_PATTERNS = [
    r"(תשאיר|השאיר|תכתוב|כתוב|תשלח|שלח|תלחץ|לחץ|תירשם|הירשם)",  # Action verbs
    r"(קישור|לינק|פרטים|ביו|בתגובות)",  # Link/bio/comments
    r"(עכשיו|היום|מחר)",  # Urgency
    r"(חינם|בחינם|ללא\s*עלות)",  # Free offer
]

PROOF_PATTERNS = [
    r"\d+%",                    # Percentage
    r"(לקוחות|עסקים|חברות)\s+\d+",  # N clients/businesses
    r"(חודש|שנה|שבוע)\s+\d+",       # Time periods with numbers
    r"(תוצאה|הצלחה|חיסכון)",         # Result words
]

PROBLEM_PATTERNS = [
    r"(בעיה|כאב|קושי|מתמודד|מתוסכל|לא\s+עובד|לא\s+מצליח)",
    r"(גונב|מבזבז|מפסיד|בזבוז)",
    r"(ידני|ידנית|לבד|עצמך)",
]

SOLUTION_PATTERNS = [
    r"(אנחנו|אני)\s+(בונ|לוקח|עוש|יוצר|מפתח|מאפשר)",
    r"(מערכת|פתרון|שירות|כלי|תהליך)",
    r"(אוטומט|אוטומציה|AI|בינה)",
]


def assign_tags(text: str) -> list:
    """Assign structural tags to a sentence based on pattern matching."""
    tags = []

    for pattern in HOOK_PATTERNS:
        if re.search(pattern, text):
            tags.append("HOOK")
            break

    for pattern in CTA_PATTERNS:
        if re.search(pattern, text):
            tags.append("CTA")
            break

    for pattern in PROOF_PATTERNS:
        if re.search(pattern, text):
            tags.append("PROOF")
            break

    for pattern in PROBLEM_PATTERNS:
        if re.search(pattern, text):
            tags.append("PROBLEM")
            break

    for pattern in SOLUTION_PATTERNS:
        if re.search(pattern, text):
            tags.append("SOLUTION")
            break

    return tags


# ── Sentence Merging (short adjacent sentences) ────

def try_merge_adjacent(sentences_in_take: list, min_words: int) -> list:
    """
    Try to merge short adjacent sentences from the same take into longer ones.
    If two consecutive sentences each have < min_words*2 and together make sense,
    offer the merged version as an additional option.

    Returns the original sentences + any merged additions.
    """
    result = list(sentences_in_take)

    for i in range(len(sentences_in_take) - 1):
        s1 = sentences_in_take[i]
        s2 = sentences_in_take[i + 1]

        # Only merge if both are short-ish and gap between them is small
        if (s1["word_count"] <= min_words * 3 and
            s2["word_count"] <= min_words * 3 and
            s2["words"][0]["start"] - s1["words"][-1]["end"] < 1.0):

            merged_words = s1["words"] + s2["words"]
            merged_text = " ".join(w["word"] for w in merged_words)
            merged_scores = score_sentence(merged_words)

            # Only keep merged if it scores better than both parts
            if merged_scores["total"] >= max(s1["score"], s2["score"]) - 0.05:
                result.append({
                    "words": merged_words,
                    "text": merged_text,
                    "word_ids": [w["id"] for w in merged_words],
                    "take_id": s1["take_id"],
                    "start": merged_words[0]["start"],
                    "end": merged_words[-1]["end"],
                    "word_count": len(merged_words),
                    "score": merged_scores["total"],
                    "scores": merged_scores,
                    "is_merged": True,
                    "merged_from": [s1.get("_idx"), s2.get("_idx")],
                })

    return result


# ── Main Logic ─────────────────────────────────────

def build_sentence_menu(merged_path: str, take_decisions_path: str,
                        similarity_threshold: float, min_words: int) -> dict:
    """
    Main function: build sentence menu from merged transcript.
    """
    start_time = time.time()

    # Load data
    merged = load_json(merged_path)
    words = merged["words"]
    take_decisions = load_json(take_decisions_path)

    # Build sets for filtering metadata
    hard_reject_ids = set()
    production_cue_ids = set()
    for dec in take_decisions.get("decisions", []):
        if dec.get("reason") == "hard_reject":
            for wid in dec["ids"]:
                hard_reject_ids.add(wid)
        elif dec.get("reason") == "production_cue":
            for wid in dec["ids"]:
                production_cue_ids.add(wid)

    # Note: we do NOT filter by remove_ids — those are metadata/hints only.
    # The sentence builder preserves all presenter words.

    # Step 1: Filter to presenter + uncertain words only (exclude reject)
    presenter_words = []
    for w in words:
        wid = w["id"]

        # Skip hard rejections and production cues
        if wid in hard_reject_ids or wid in production_cue_ids:
            continue

        # Skip fillers (marked by filler_detector)
        if w.get("is_filler", False):
            continue

        # Keep presenter and uncertain words
        if w.get("final_decision") in ("presenter", "uncertain") or w.get("is_presenter", False):
            presenter_words.append(w)

    print(f"[sentence_builder] Presenter words: {len(presenter_words)} / {len(words)} total",
          file=sys.stderr)

    # Step 2: Group by take_id
    takes = defaultdict(list)
    for w in presenter_words:
        tid = w.get("take_id", 0)
        takes[tid].append(w)

    # Sort words within each take by start time
    for tid in takes:
        takes[tid].sort(key=lambda w: w["start"])

    print(f"[sentence_builder] Takes: {len(takes)}", file=sys.stderr)

    # Step 3: Split each take into sentences
    all_sentences = []
    sentence_idx = 0

    for tid in sorted(takes.keys()):
        take_words = takes[tid]
        raw_sentences = split_take_to_sentences(take_words)

        take_sentences = []
        for sent_words in raw_sentences:
            text = " ".join(w["word"] for w in sent_words)

            # Filter: production cues
            if is_production_cue(text):
                continue

            # Filter: filler-only sentences
            if is_filler_only(sent_words):
                continue

            # Filter: too short
            if len(sent_words) < min_words:
                continue

            scores = score_sentence(sent_words)
            tags = assign_tags(text)

            sent_obj = {
                "_idx": sentence_idx,
                "words": sent_words,
                "text": text,
                "word_ids": [w["id"] for w in sent_words],
                "take_id": tid,
                "start": sent_words[0]["start"],
                "end": sent_words[-1]["end"],
                "word_count": len(sent_words),
                "score": scores["total"],
                "scores": scores,
                "tags": tags,
                "has_forbidden_start": has_forbidden_start(sent_words),
                "is_merged": False,
            }

            take_sentences.append(sent_obj)
            sentence_idx += 1

        # Step 3.5: Try merging short adjacent sentences
        take_sentences_with_merged = try_merge_adjacent(take_sentences, min_words)
        for s in take_sentences_with_merged:
            if s.get("is_merged") and "_idx" not in s:
                s["_idx"] = sentence_idx
                sentence_idx += 1

        all_sentences.extend(take_sentences_with_merged)

    print(f"[sentence_builder] Total sentences (before dedup): {len(all_sentences)}",
          file=sys.stderr)

    # Step 4: Group similar sentences
    groups = group_similar_sentences(all_sentences, similarity_threshold)

    # Step 5: For each group, pick the best version
    output_sentences = []
    output_groups = []
    sentence_counter = 1

    for group_idx, group_indices in enumerate(groups):
        group_sents = [all_sentences[i] for i in group_indices]

        # Sort by score descending, then by take_id descending (prefer later takes)
        group_sents.sort(key=lambda s: (s["score"], s["take_id"]), reverse=True)

        best = group_sents[0]
        group_id = f"G{group_idx + 1}"

        group_sentence_ids = []
        for rank, sent in enumerate(group_sents):
            sid = f"S{sentence_counter}"
            sentence_counter += 1

            is_best = (rank == 0)

            output_sentences.append({
                "id": sid,
                "text": sent["text"],
                "word_ids": sent["word_ids"],
                "take_id": sent["take_id"],
                "start": round(sent["start"], 3),
                "end": round(sent["end"], 3),
                "word_count": sent["word_count"],
                "score": sent["score"],
                "scores": sent["scores"],
                "group_id": group_id,
                "is_best_in_group": is_best,
                "alternatives": len(group_sents) - 1,
                "tags": sent.get("tags", []),
                "has_forbidden_start": sent.get("has_forbidden_start", False),
                "is_merged": sent.get("is_merged", False),
            })

            group_sentence_ids.append(sid)

        output_groups.append({
            "group_id": group_id,
            "representative_text": best["text"][:80] + ("..." if len(best["text"]) > 80 else ""),
            "version_count": len(group_sents),
            "best_sentence_id": group_sentence_ids[0],
            "all_sentence_ids": group_sentence_ids,
        })

    # Sort sentences by their time position (start time of best version per group)
    # But keep the sentence IDs stable
    output_sentences.sort(key=lambda s: (not s["is_best_in_group"], s["start"]))

    # Stats
    processing_time_ms = int((time.time() - start_time) * 1000)

    best_sentences = [s for s in output_sentences if s["is_best_in_group"]]
    total_best_words = sum(s["word_count"] for s in best_sentences)
    coverage = total_best_words / len(presenter_words) if presenter_words else 0

    # Tag distribution
    tag_counts = defaultdict(int)
    for s in best_sentences:
        for tag in s.get("tags", []):
            tag_counts[tag] += 1

    stats = {
        "total_words_in": len(words),
        "presenter_words": len(presenter_words),
        "total_sentences": len(output_sentences),
        "unique_groups": len(output_groups),
        "best_sentences": len(best_sentences),
        "best_sentences_total_words": total_best_words,
        "coverage_of_presenter": round(coverage, 3),
        "sentences_with_forbidden_start": sum(1 for s in output_sentences if s["has_forbidden_start"]),
        "merged_sentences": sum(1 for s in output_sentences if s["is_merged"]),
        "tag_distribution": dict(tag_counts),
        "processing_time_ms": processing_time_ms,
    }

    return {
        "sentences": output_sentences,
        "groups": output_groups,
        "stats": stats,
    }


def format_menu_for_ai(menu: dict) -> str:
    """
    Format the sentence menu as text that the AI can read and select from.
    Only includes best-in-group sentences for the primary menu,
    with alternatives noted.
    """
    lines = ["SENTENCE MENU:", ""]

    best_sentences = [s for s in menu["sentences"] if s["is_best_in_group"]]
    # Sort by start time for chronological presentation
    best_sentences.sort(key=lambda s: s["start"])

    for s in best_sentences:
        tags_str = f" [{', '.join(s['tags'])}]" if s.get("tags") else ""
        alt_str = f", {s['alternatives']} alternatives" if s["alternatives"] > 0 else ""
        warn_str = " ⚠ FORBIDDEN_START" if s.get("has_forbidden_start") else ""
        merged_str = " [MERGED]" if s.get("is_merged") else ""

        lines.append(
            f'{s["id"]} (score {s["score"]}, {s["word_count"]} words, '
            f'take {s["take_id"]}{alt_str}{tags_str}{warn_str}{merged_str}): '
            f'"{s["text"]}"'
        )

    lines.append("")
    lines.append(f"Total: {len(best_sentences)} sentences available")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────

def main():
    args = parse_args()

    menu = build_sentence_menu(
        merged_path=args.merged,
        take_decisions_path=args.take_decisions,
        similarity_threshold=args.similarity_threshold,
        min_words=args.min_words,
    )

    # Write full menu to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(menu, f, ensure_ascii=False, indent=2)

    # Write to stdout for pipeline consumption
    json.dump(menu, sys.stdout, ensure_ascii=False, indent=2)

    # Print summary
    stats = menu["stats"]
    print(f"\n[sentence_builder] Done: "
          f"{stats['unique_groups']} groups, "
          f"{stats['best_sentences']} best sentences, "
          f"{stats['best_sentences_total_words']} words, "
          f"coverage {stats['coverage_of_presenter']:.0%}, "
          f"{stats['processing_time_ms']}ms",
          file=sys.stderr)

    # Also print AI-readable menu to stderr for debugging
    print("\n" + format_menu_for_ai(menu), file=sys.stderr)


if __name__ == "__main__":
    main()

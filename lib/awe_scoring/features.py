from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*", re.UNICODE)
SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
FORBIDDEN_NAME_PARTS = ("expert", "item", "key", "label", "prompt", "response", "selection", "slug", "taqr")

CONNECTIVES = {
    "en": {"because", "therefore", "however", "first", "second", "finally", "although", "so", "thus", "consequently"},
    "fr": {"parce", "donc", "cependant", "premièrement", "deuxièmement", "finalement", "bien", "ainsi", "conséquent"},
}
EXAMPLE_MARKERS = {"en": {"example", "instance", "such"}, "fr": {"exemple", "comme", "notamment"}}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _safe_stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def portable_features(text: str, language: str) -> dict[str, float]:
    words = TOKEN.findall(text)
    lowered = [word.casefold() for word in words]
    counts = Counter(lowered)
    sentences = [part.strip() for part in SENTENCE.findall(text) if TOKEN.search(part)]
    sentence_lengths = [len(TOKEN.findall(sentence)) for sentence in sentences]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    alpha = sum(character.isalpha() for character in text)
    terminal = sum(sentence.rstrip().endswith((".", "!", "?")) for sentence in sentences)
    repeated_tokens = sum(max(0, count - 1) for count in counts.values())
    long_words = sum(len(word) >= 8 for word in words)
    comma = text.count(",")
    semicolon = text.count(";")
    repeated_punctuation = len(re.findall(r"[!?.,]{2,}", text))
    lowercase_starts = sum(bool(re.match(r"^[a-zà-öø-ÿ]", sentence)) for sentence in sentences)
    possible_fragments = sum(length <= 3 for length in sentence_lengths)
    possible_run_ons = sum(length >= 40 for length in sentence_lengths)
    connective_count = sum(word in CONNECTIVES[language] for word in lowered)
    example_count = sum(word in EXAMPLE_MARKERS[language] for word in lowered)
    features = {
        "det_portable_word_count": len(words),
        "det_portable_character_count": len(text),
        "det_portable_alphabetic_character_count": alpha,
        "det_portable_sentence_count": len(sentences),
        "det_portable_paragraph_count": len(paragraphs),
        "det_portable_average_words_per_sentence": _ratio(sum(sentence_lengths), len(sentence_lengths)),
        "det_portable_maximum_words_in_sentence": max(sentence_lengths, default=0),
        "det_portable_sentence_length_stdev": _safe_stdev([float(value) for value in sentence_lengths]),
        "det_portable_terminal_punctuation_ratio": _ratio(terminal, len(sentences)),
        "det_portable_lowercase_sentence_start_count": lowercase_starts,
        "det_portable_possible_short_fragment_count": possible_fragments,
        "det_portable_possible_run_on_count": possible_run_ons,
        "det_portable_repeated_punctuation_count": repeated_punctuation,
        "det_portable_comma_per_100_words": 100 * _ratio(comma, len(words)),
        "det_portable_semicolon_per_100_words": 100 * _ratio(semicolon, len(words)),
        "det_portable_unique_token_ratio": _ratio(len(counts), len(words)),
        "det_portable_hapax_ratio": _ratio(sum(count == 1 for count in counts.values()), len(counts)),
        "det_portable_repeated_token_ratio": _ratio(repeated_tokens, len(words)),
        "det_portable_average_word_length": _ratio(sum(len(word) for word in words), len(words)),
        "det_portable_long_word_ratio": _ratio(long_words, len(words)),
        "det_portable_connectives_per_100_words": 100 * _ratio(connective_count, len(words)),
        "det_portable_example_markers_per_100_words": 100 * _ratio(example_count, len(words)),
        "det_portable_cv_under_30_words": float(len(words) < 30),
        "det_portable_is_blank_or_nonlinguistic": float(not text.strip()),
    }
    return {name: float(value) for name, value in features.items()}


def validate_provided_features(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("deterministic_features must be a non-empty object")
    output: dict[str, float] = {}
    for name, raw in value.items():
        lowered = str(name).lower()
        if any(fragment in lowered for fragment in FORBIDDEN_NAME_PARTS):
            raise ValueError(f"Forbidden predictor name: {name}")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite predictor {name}={raw!r}")
        output[str(name)] = number
    return output


def deterministic_features(row: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    supplied = row.get("deterministic_features")
    if supplied and config.get("prefer_provided", True):
        return validate_provided_features(supplied)
    if config.get("mode", "portable") == "provided":
        return validate_provided_features(supplied)
    from .routing import normalize_language

    return portable_features(str(row.get("response_text", "")), normalize_language(row["language"]))


def error_dispersion(features: dict[str, float]) -> int:
    if "det_derived_d_value" in features:
        return int(max(0, min(3, round(features["det_derived_d_value"]))))
    burden = (
        features.get("det_portable_possible_short_fragment_count", 0)
        + features.get("det_portable_possible_run_on_count", 0)
        + features.get("det_portable_repeated_punctuation_count", 0)
        + features.get("det_portable_lowercase_sentence_start_count", 0)
    )
    sentences = max(1.0, features.get("det_portable_sentence_count", 1))
    density = burden / sentences
    return 3 if burden == 0 else 2 if density <= 0.25 else 1 if density <= 0.75 else 0

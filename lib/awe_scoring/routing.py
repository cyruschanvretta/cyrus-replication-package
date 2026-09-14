from __future__ import annotations

from typing import Any

RANGE_LABELS = ("Low", "Medium", "High")


def normalize_language(value: Any) -> str:
    language = str(value).strip().lower()
    aliases = {"en": "en", "english": "en", "fr": "fr", "french": "fr"}
    if language not in aliases:
        raise ValueError(f"Unsupported language: {value!r}")
    return aliases[language]


def normalize_skill(value: Any) -> str:
    skill = str(value).strip().upper()
    if skill not in {"TD", "CV"}:
        raise ValueError(f"Unsupported skill: {value!r}")
    return skill


def route_key(language: Any, skill: Any) -> str:
    return f"{'english' if normalize_language(language) == 'en' else 'french'}-{normalize_skill(skill).lower()}"


def score_values(skill: str) -> list[int]:
    return list(range(0, 7)) if normalize_skill(skill) == "TD" else list(range(0, 5))


def score_to_range(score: int | float, skill: str) -> str:
    value = float(score)
    if normalize_skill(skill) == "CV":
        if not 0 <= value <= 4:
            raise ValueError(f"CV score outside 0..4: {score}")
        return "Low" if value <= 1 else "Medium" if value <= 3 else "High"
    if not 0 <= value <= 6:
        raise ValueError(f"TD score outside 0..6: {score}")
    return "Low" if value <= 2 else "Medium" if value <= 4 else "High"


def hard_gate(row: dict[str, Any], word_count: int) -> tuple[int | None, str | None]:
    if row.get("is_linguistic") is False or not str(row.get("response_text", "")).strip():
        return 0, "blank-or-non-linguistic"
    if normalize_skill(row["skill"]) == "CV" and word_count < 30:
        return 1, "cv-under-30-authored-words"
    return None, None

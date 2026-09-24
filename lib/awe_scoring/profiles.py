from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from .config import package_root

TD_FIELDS = [
    ("T", "task engagement", ("unrelated", "partly related", "directly addresses task", "precisely frames task")),
    ("P", "position stability", ("no position", "implied or unstable", "explicit and sustained", "qualified or nuanced")),
    ("B", "distinct supporting reasons", ("none", "one", "two", "three or more differentiated reasons")),
    ("S", "reasons with concrete task-relevant detail", ("none", "one weak or generic reason", "one to two concrete reasons", "concrete detail sustained across three or more reasons")),
    ("L", "support-to-position linkage", ("detached", "mostly implied", "usually explicit", "integrated across reasons")),
    ("E", "reasons unpacked with how, why, consequence, or significance", ("none", "one", "two", "three or more")),
    ("O", "functional organization", ("disconnected", "visible but list-like", "logical relationships", "coherent progression")),
    ("A", "idea advancement", ("repeats", "mostly parallel additions", "later ideas advance", "later ideas build or synthesize")),
    ("F", "functional completeness", ("fragment or list", "partial response", "opening and development roles", "opening, development, and closure roles")),
    ("C", "internal consistency", ("contradictory", "uneven", "maintained", "handles qualification or counterpoint")),
]

CV_MODEL_FIELDS = [
    ("I", "ease of understanding on an ordinary read", ("central meaning blocked", "repeated rereading", "brief local distraction", "uninterrupted")),
    ("R", "repair ease and localizability", ("whole-message reconstruction", "several clauses or sentences rebuilt", "local edits preserve meaning", "no or trivial mental repair")),
    ("G", "grammar control", ("pervasive failure", "frequent failure", "mostly controlled", "stable control")),
    ("S", "sentence-boundary and syntax control", ("obscuring", "unstable", "mostly controlled", "varied and controlled")),
    ("U", "usage control", ("pervasive failure", "frequent failure", "mostly controlled", "stable control")),
    ("P", "punctuation control", ("pervasive failure", "frequent failure", "mostly controlled", "stable control")),
    ("O", "orthographic control", ("pervasive failure", "frequent failure", "mostly controlled", "stable control")),
    ("A", "syntactic ambition", ("fragments", "simple and repetitive", "mixed structures", "varied complex structures")),
    ("C", "control consistency", ("unstable throughout", "uneven", "mostly stable", "stable throughout")),
]

# Compact LLM profile field sets a route may request through `profile_schema`.
PROFILE_SCHEMAS = ("TD", "CV")


def _read_context(name: str) -> str:
    path = package_root() / "context_materials/profiles" / name
    return path.read_text(encoding="utf-8").strip()


def template(skill: str) -> str:
    fields = TD_FIELDS if skill.upper() == "TD" else CV_MODEL_FIELDS
    return "|".join(f"{code}#" for code, _, _ in fields)


def instructions(skill: str) -> str:
    fields = TD_FIELDS if skill.upper() == "TD" else CV_MODEL_FIELDS
    order = "".join(code for code, _, _ in fields)
    lines = [
        "Describe observable writing only. Do not assign a rubric band or numeric score.",
        "All fields use the same polarity: 0 is least/weakest and 3 is most/strongest.",
        "Begin at 0. Raise a field only after locating evidence for every lower threshold.",
        "A 3 requires the stated strongest property across the response; it is not a default for length or relevance.",
        f"Return exactly one labeled line in this order: {template(skill)}",
        f"Replace each # with one digit 0-3. Required field-order checksum: {order}. No prose, JSON, markdown, or spaces.",
    ]
    if skill.upper() == "CV":
        lines.append("Error dispersion D is supplied by deterministic software; do not emit D.")
    lines.extend(
        f"{code} {name}: " + "; ".join(f"{index}={label}" for index, label in enumerate(levels))
        for code, name, levels in fields
    )
    return "\n".join(lines)


def build_messages(
    row: dict[str, Any], deterministic: dict[str, float], route: dict[str, Any], profile_schema: str
) -> list[dict[str, str]]:
    skill = profile_schema.upper()
    context_names = ["first-principles-core.md", "descriptive-writing-profile.md"]
    context_names.append("td-first-principles.md" if skill == "TD" else "cv-first-principles.md")
    system = "\n\n".join([
        f"You are a {skill} writing-description classifier, not a score reporter.",
        *[_read_context(name) for name in context_names],
        "FINAL OUTPUT CONTRACT:\n" + instructions(skill),
    ])
    parts: list[str] = []
    if skill == "TD" and not route.get("omit_task_prompt", False):
        task = str(row.get("task_prompt", "")).strip()
        if not task:
            raise ValueError(f"TD response {row['id']} requires task_prompt for the LLM profile route")
        parts.append(f"CURRENT TASK:\n<task>\n{task}\n</task>")
    parts.append("DETERMINISTIC RESPONSE EVIDENCE:\n" + json.dumps(deterministic, sort_keys=True, separators=(",", ":")))
    parts.append(f"CURRENT STUDENT RESPONSE:\n<response>\n{row['response_text']}\n</response>")
    parts.append(f"OUTPUT NOW: copy this template, replace each # with one digit 0-3, and output nothing else:\n{template(skill)}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n---\n\n".join(parts)}]


def parse_profile(content: str, skill: str, derived_d: int | None = None) -> dict[str, int]:
    fields = TD_FIELDS if skill.upper() == "TD" else CV_MODEL_FIELDS
    required = [field[0] for field in fields]
    first_line = str(content).strip().splitlines()[0] if str(content).strip() else ""
    tokens = first_line.split("|")
    values: dict[str, int] = {}
    if len(tokens) == len(required):
        for expected, token in zip(required, tokens):
            match = re.fullmatch(rf"{expected}[:=]?([0-3])", token.strip(), re.IGNORECASE)
            if not match:
                break
            values[expected] = int(match.group(1))
    if len(values) != len(required):
        compact = re.sub(r"[\s|,;:=_\-]", "", first_line)
        if re.fullmatch(rf"[0-3]{{{len(required)}}}", compact):
            values = dict(zip(required, map(int, compact)))
    if len(values) != len(required):
        raise ValueError(f"Invalid {skill} compact profile: {first_line[:160]!r}")
    if skill.upper() == "CV":
        if derived_d is None or not 0 <= derived_d <= 3:
            raise ValueError("CV profile requires deterministic D in 0..3")
        values["D"] = int(derived_d)
    return values


def provisional_score(skill: str, values: dict[str, int]) -> int:
    if skill.upper() == "TD":
        thresholds = [
            values["T"] >= 1 and values["P"] >= 1 and values["B"] >= 1,
            values["T"] >= 2 and values["P"] >= 1 and values["B"] >= 1 and (values["S"] >= 1 or values["E"] >= 1),
            values["T"] >= 2 and values["P"] >= 2 and values["B"] >= 2 and values["S"] >= 1 and values["L"] >= 2 and values["E"] >= 1,
            values["P"] >= 2 and values["B"] >= 2 and values["S"] >= 2 and values["L"] >= 2 and values["E"] >= 2 and values["O"] >= 2 and values["C"] >= 2,
            values["P"] >= 3 and values["B"] >= 3 and values["S"] >= 3 and values["L"] >= 3 and values["E"] >= 3 and values["O"] >= 3 and values["A"] >= 3 and values["C"] >= 3,
        ]
        score = 1
        for passed in thresholds:
            if not passed:
                break
            score += 1
        return score
    weights = {"I": 3, "R": 2, "G": 1, "S": 1, "U": 1, "P": 1, "O": 1, "D": 1.5, "C": 1.5}
    average = sum(values[key] * weight for key, weight in weights.items()) / sum(weights.values())
    score = max(1, min(4, 1 + round(average)))
    if values["I"] == 0 or values["R"] == 0:
        score = 1
    elif values["I"] == 1:
        score = min(score, 2)
    return score


def profile_features(skill: str, values: dict[str, int]) -> dict[str, float]:
    # The qwen_ prefix is retained solely for frozen feature-contract parity;
    # values may come from Gemma or another supplied model.
    output: dict[str, float] = {"qwen_profile_present": 1.0}
    numeric = [float(value) for value in values.values()]
    for code, value in sorted(values.items()):
        output[f"qwen_{code.lower()}"] = float(value)
        for threshold in (1, 2, 3):
            output[f"qwen_{code.lower()}_ge_{threshold}"] = float(value >= threshold)
    output["qwen_provisional_score"] = float(provisional_score(skill, values))
    output["qwen_agg_mean"] = statistics.fmean(numeric)
    output["qwen_agg_stdev"] = statistics.pstdev(numeric)
    output["qwen_agg_min"] = min(numeric)
    output["qwen_agg_max"] = max(numeric)
    for threshold in (1, 2, 3):
        output[f"qwen_agg_count_ge_{threshold}"] = float(sum(value >= threshold for value in numeric))
    if skill.upper() == "CV":
        reader = [values[code] for code in ("I", "R")]
        controls = [values[code] for code in ("G", "S", "U", "P", "O", "D", "C")]
        output.update({
            "qwen_agg_reader_mean": statistics.fmean(reader),
            "qwen_agg_reader_min": float(min(reader)),
            "qwen_agg_control_mean": statistics.fmean(controls),
            "qwen_agg_control_min": float(min(controls)),
            "qwen_agg_control_ge_2_count": float(sum(value >= 2 for value in controls)),
        })
    else:
        support = [values[code] for code in ("B", "S", "L", "E")]
        discourse = [values[code] for code in ("O", "A", "F", "C")]
        output.update({
            "qwen_agg_support_mean": statistics.fmean(support),
            "qwen_agg_support_min": float(min(support)),
            "qwen_agg_discourse_mean": statistics.fmean(discourse),
            "qwen_agg_discourse_min": float(min(discourse)),
        })
    return output

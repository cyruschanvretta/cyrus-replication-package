from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .estimators import LEARNERS
from .profiles import PROFILE_SCHEMAS

_MISSING = object()
_SOURCES = ("field", "feature", "measure")
_MEASURES = ("word_count",)
_COMPARISONS = ("lt", "le", "gt", "ge")
_OPERATORS = ("equals", "not_equals", "in", "blank", *_COMPARISONS)
_FEATURE_VIEWS = {"deterministic": False, "combined": True}


def normalize_language(value: Any) -> str:
    language = str(value).strip().lower()
    aliases = {"en": "en", "english": "en", "fr": "fr", "french": "fr"}
    if language not in aliases:
        raise ValueError(f"Unsupported language: {value!r}")
    return aliases[language]


def normalize_skill(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip().upper()


@dataclass(frozen=True)
class Skill:
    name: str
    score_values: tuple[int, ...]
    ranges: tuple[tuple[str, tuple[int, ...]], ...]
    ungated_min_score: int | None = None

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.ranges]

    def to_range(self, score: int | float) -> str:
        value = float(score)
        low, high = self.score_values[0], self.score_values[-1]
        if not low <= value <= high or (value.is_integer() and int(value) not in self.score_values):
            raise ValueError(f"{self.name} score outside {low}..{high}: {score}")
        for label, members in self.ranges:
            if value <= members[-1]:
                return label
        raise AssertionError("ranges cover every score value")


@dataclass(frozen=True)
class HardGate:
    name: str
    score: int
    when: dict[str, Any]
    skills: frozenset[str] | None = None
    routes: frozenset[str] | None = None

    def applies_to(self, route: "Route") -> bool:
        if self.skills is not None and route.skill.name not in self.skills:
            return False
        return self.routes is None or route.name in self.routes

    def matches(self, row: dict[str, Any], features: dict[str, float], word_count: int) -> bool:
        return _evaluate(self.when, row, features, {"word_count": word_count})


@dataclass(frozen=True)
class Route:
    name: str
    language: str
    skill: Skill
    learner: str
    uses_llm: bool
    profile_schema: str | None
    settings: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ScoringSpec:
    skills: dict[str, Skill]
    routes: dict[str, Route]
    hard_gates: tuple[HardGate, ...]


def _same(left: Any, right: Any) -> bool:
    # Booleans compare by identity so that 0/1 never stand in for false/true.
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


def _evaluate(condition: dict[str, Any], row: dict[str, Any], features: dict[str, float], measures: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(_evaluate(part, row, features, measures) for part in condition["all"])
    if "any" in condition:
        return any(_evaluate(part, row, features, measures) for part in condition["any"])
    if "field" in condition:
        value = row.get(condition["field"], _MISSING)
    elif "feature" in condition:
        value = features.get(condition["feature"], _MISSING)
    else:
        value = measures.get(condition["measure"], _MISSING)
    if "blank" in condition:
        text = "" if value is _MISSING or value is None else str(value)
        return (not text.strip()) == bool(condition["blank"])
    if value is _MISSING:
        return False
    if "equals" in condition:
        return _same(value, condition["equals"])
    if "not_equals" in condition:
        return not _same(value, condition["not_equals"])
    if "in" in condition:
        return any(_same(value, option) for option in condition["in"])
    number = float(value)
    for name, compare in (("lt", float.__lt__), ("le", float.__le__), ("gt", float.__gt__), ("ge", float.__ge__)):
        if name in condition:
            return compare(number, float(condition[name]))
    raise AssertionError("conditions are validated at compile time")


def _validate_condition(condition: Any, where: str) -> None:
    if not isinstance(condition, dict):
        raise ValueError(f"{where}: condition must be a mapping, got {condition!r}")
    for group in ("all", "any"):
        if group in condition:
            parts = condition[group]
            if len(condition) != 1 or not isinstance(parts, list) or not parts:
                raise ValueError(f"{where}: '{group}' must be the only key and hold a non-empty list")
            for index, part in enumerate(parts):
                _validate_condition(part, f"{where}.{group}[{index}]")
            return
    sources = [key for key in _SOURCES if key in condition]
    operators = [key for key in _OPERATORS if key in condition]
    unknown = sorted(set(condition) - set(_SOURCES) - set(_OPERATORS))
    if len(sources) != 1 or len(operators) != 1 or unknown:
        raise ValueError(
            f"{where}: condition needs exactly one of {list(_SOURCES)} and one of {list(_OPERATORS)}; got {condition!r}"
        )
    if "measure" in condition and condition["measure"] not in _MEASURES:
        raise ValueError(f"{where}: unknown measure {condition['measure']!r}; available: {list(_MEASURES)}")
    operator = operators[0]
    if operator == "in" and not isinstance(condition["in"], list):
        raise ValueError(f"{where}: 'in' requires a list")
    if operator in _COMPARISONS and not isinstance(condition[operator], (int, float)):
        raise ValueError(f"{where}: '{operator}' requires a number")


def _compile_skill(name: str, spec: Any) -> Skill:
    if not isinstance(spec, dict) or not spec.get("score_values"):
        raise ValueError(f"skills.{name} requires a non-empty score_values list")
    values = [int(value) for value in spec["score_values"]]
    if values != sorted(set(values)):
        raise ValueError(f"skills.{name}.score_values must be unique and ascending: {spec['score_values']}")
    raw_ranges = spec.get("ranges")
    if raw_ranges is None:
        ranges = [(str(value), (value,)) for value in values]
    else:
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise ValueError(f"skills.{name}.ranges must be a non-empty list")
        ranges = []
        previous_max = float("-inf")
        for index, entry in enumerate(raw_ranges):
            where = f"skills.{name}.ranges[{index}]"
            if not isinstance(entry, dict) or not str(entry.get("label", "")).strip():
                raise ValueError(f"{where} requires a label")
            if ("max" in entry) == ("scores" in entry):
                raise ValueError(f"{where} requires exactly one of 'max' or 'scores'")
            if "max" in entry:
                members = tuple(value for value in values if previous_max < value <= entry["max"])
            else:
                members = tuple(sorted(int(value) for value in entry["scores"]))
            if not members:
                raise ValueError(f"{where} ({entry['label']}) covers no score values")
            ranges.append((str(entry["label"]), members))
            previous_max = max(previous_max, members[-1])
    labels = [label for label, _ in ranges]
    if len(labels) != len(set(labels)):
        raise ValueError(f"skills.{name}.ranges labels must be unique: {labels}")
    flattened = [value for _, members in ranges for value in members]
    if flattened != values:
        raise ValueError(
            f"skills.{name}.ranges must cover every score value exactly once, in ascending order; "
            f"score_values={values}, ranges={dict(ranges)}"
        )
    floor = spec.get("ungated_min_score")
    if floor is not None and int(floor) not in values:
        raise ValueError(f"skills.{name}.ungated_min_score {floor} is not a score value")
    return Skill(name, tuple(values), tuple(ranges), None if floor is None else int(floor))


def _compile_route(name: str, settings: Any, skills: dict[str, Skill]) -> Route:
    if not isinstance(settings, dict):
        raise ValueError(f"routes.{name} must be a mapping")
    for key in ("language", "skill", "learner", "uses_llm"):
        if key not in settings:
            raise ValueError(f"routes.{name} requires '{key}'")
    skill_name = normalize_skill(settings["skill"])
    if skill_name not in skills:
        raise ValueError(f"routes.{name}: skill {settings['skill']!r} is not defined under 'skills' ({sorted(skills)})")
    learner = settings["learner"]
    if learner not in LEARNERS:
        raise ValueError(f"routes.{name}: unknown learner {learner!r}; available: {list(LEARNERS)}")
    uses_llm = settings["uses_llm"]
    if not isinstance(uses_llm, bool):
        raise ValueError(f"routes.{name}.uses_llm must be true or false")
    view = settings.get("feature_view")
    if view is not None:
        if view not in _FEATURE_VIEWS:
            raise ValueError(f"routes.{name}.feature_view must be one of {list(_FEATURE_VIEWS)}")
        if _FEATURE_VIEWS[view] != uses_llm:
            raise ValueError(f"routes.{name}: feature_view {view!r} is inconsistent with uses_llm={uses_llm}")
    schema = None
    if uses_llm:
        schema = normalize_skill(settings.get("profile_schema") or skill_name)
        if schema not in PROFILE_SCHEMAS:
            raise ValueError(
                f"routes.{name} uses the LLM but has no supported profile_schema "
                f"(got {schema!r}; available: {list(PROFILE_SCHEMAS)})"
            )
    return Route(name, normalize_language(settings["language"]), skills[skill_name], learner, uses_llm, schema, settings)


def _compile_gate(index: int, spec: Any, skills: dict[str, Skill], routes: dict[str, Route]) -> HardGate:
    where = f"hard_gates[{index}]"
    if not isinstance(spec, dict) or not spec.get("name") or "score" not in spec or "when" not in spec:
        raise ValueError(f"{where} requires name, score, and when")
    _validate_condition(spec["when"], f"{where}.when")
    gate_skills = None if spec.get("skills") is None else frozenset(normalize_skill(value) for value in spec["skills"])
    gate_routes = None if spec.get("routes") is None else frozenset(str(value) for value in spec["routes"])
    for value in gate_skills or ():
        if value not in skills:
            raise ValueError(f"{where} refers to undefined skill {value!r}")
    for value in gate_routes or ():
        if value not in routes:
            raise ValueError(f"{where} refers to undefined route {value!r}")
    gate = HardGate(str(spec["name"]), int(spec["score"]), spec["when"], gate_skills, gate_routes)
    for route in routes.values():
        if gate.applies_to(route) and gate.score not in route.skill.score_values:
            raise ValueError(f"{where} score {gate.score} is not a {route.skill.name} score value (route {route.name})")
    return gate


def compile_spec(config: dict[str, Any]) -> ScoringSpec:
    raw_skills = config.get("skills")
    if not isinstance(raw_skills, dict) or not raw_skills:
        raise ValueError("config requires a non-empty 'skills' section")
    skills: dict[str, Skill] = {}
    for name, spec in raw_skills.items():
        key = normalize_skill(name)
        if key is None or key in skills:
            raise ValueError(f"skills has an empty or duplicate name: {name!r}")
        skills[key] = _compile_skill(key, spec)
    raw_routes = config.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError("config requires a non-empty 'routes' section")
    routes = {str(name): _compile_route(str(name), settings, skills) for name, settings in raw_routes.items()}
    selectors: dict[tuple[str, str], str] = {}
    for route in routes.values():
        selector = (route.language, route.skill.name)
        if selector in selectors:
            raise ValueError(f"routes {selectors[selector]} and {route.name} both select language={selector[0]}, skill={selector[1]}")
        selectors[selector] = route.name
    raw_gates = config.get("hard_gates") or []
    if not isinstance(raw_gates, list):
        raise ValueError("hard_gates must be a list")
    gates = tuple(_compile_gate(index, spec, skills, routes) for index, spec in enumerate(raw_gates))
    return ScoringSpec(skills, routes, gates)


def resolve_route(spec: ScoringSpec, row: dict[str, Any]) -> Route:
    language = normalize_language(row["language"])
    skill = normalize_skill(row.get("skill"))
    if skill is not None and skill not in spec.skills:
        raise ValueError(f"Unsupported skill: {row.get('skill')!r}")
    matches = [
        route for route in spec.routes.values()
        if route.language == language and (skill is None or route.skill.name == skill)
    ]
    if not matches:
        raise ValueError(f"No route for language={language!r}, skill={skill!r}")
    if len(matches) > 1:
        raise ValueError(f"Row {row.get('id')!r} needs a skill to choose between routes {[route.name for route in matches]}")
    return matches[0]


def hard_gate(
    spec: ScoringSpec,
    route: Route,
    row: dict[str, Any],
    features: dict[str, float],
    word_count: int,
) -> tuple[int | None, str | None]:
    for gate in spec.hard_gates:
        if gate.applies_to(route) and gate.matches(row, features, word_count):
            return gate.score, gate.name
    return None, None

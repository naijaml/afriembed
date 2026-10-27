"""Language registry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import load_yaml


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str
    tier: str
    fleurs: str | None
    sib: str | None
    africaption: str | None
    injongo: str | None
    lite: bool = False


def _registry() -> dict:
    return load_yaml("languages.yaml")


def all_languages() -> dict[str, Language]:
    out = {}
    for code, d in _registry()["languages"].items():
        out[code] = Language(
            code=code,
            lite=bool(d.get("lite", False)),
            **{k: d.get(k) for k in ("name", "script", "tier", "fleurs", "sib", "africaption", "injongo")},
        )
    return out


def get(code: str) -> Language:
    langs = all_languages()
    if code not in langs:
        raise KeyError(f"Unknown language code {code!r}. Known: {sorted(langs)}")
    return langs[code]


def language_set(name: str) -> list[str]:
    sets = _registry()["sets"]
    if name not in sets:
        raise KeyError(f"Unknown language set {name!r}. Known: {sorted(sets)}")
    return list(sets[name])


def resolve(spec: str | None, default_set: str) -> list[str]:
    """Turn 'lite' / 'hau,yor,swh' / None into a list of ISO codes."""
    if not spec:
        return language_set(default_set)
    if spec in _registry()["sets"]:
        return language_set(spec)
    codes = [c.strip() for c in spec.split(",") if c.strip()]
    for c in codes:
        get(c)
    return codes


def with_fleurs(codes: list[str]) -> list[Language]:
    langs = [get(c) for c in codes]
    missing = [l.code for l in langs if not l.fleurs]
    if missing:
        import warnings

        warnings.warn(f"Languages without FLEURS audio will be skipped: {missing}", stacklevel=2)
    return [l for l in langs if l.fleurs]

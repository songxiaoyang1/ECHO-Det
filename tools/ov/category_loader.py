import json
from json import JSONDecodeError
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def _normalize_names(names):
    if isinstance(names, Mapping):
        return [
            {"id": int(idx), "name": str(name)}
            for idx, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [{"id": idx, "name": str(name)} for idx, name in enumerate(names)]
    raise ValueError(f"names must be a mapping or sequence, got {type(names)!r}")


def _load_json_categories(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError(f"No categories or names found in {path}")
    return [
        {"id": int(item["id"]), "name": str(item["name"])}
        for item in sorted(categories, key=lambda item: int(item["id"]))
    ]


def _load_yaml_categories(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names") if isinstance(data, Mapping) else None
    if not names:
        raise ValueError(f"No categories or names found in {path}")
    return _normalize_names(names)


def load_categories(path: str | Path):
    path = Path(path)
    try:
        return _load_json_categories(path)
    except JSONDecodeError:
        pass
    return _load_yaml_categories(path)

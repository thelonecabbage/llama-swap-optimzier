"""Load local benchmark configuration without third-party dependencies."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


ENV_PATH = Path(__file__).with_name(".env")
ASSIGNMENT_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class EnvConfigError(ValueError):
    pass


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EnvConfigError(f"cannot read environment file {path}: {error}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assignment = ASSIGNMENT_RE.fullmatch(line)
        if assignment is None:
            raise EnvConfigError(f"invalid assignment at {path}:{line_number}")
        name, value = assignment.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def load_environment(
    path: Path = ENV_PATH,
    process_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = parse_env_file(path)
    environment.update(process_environment if process_environment is not None else os.environ)
    return environment
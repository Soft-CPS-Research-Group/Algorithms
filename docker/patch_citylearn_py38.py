from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


FUTURE_IMPORT = "from __future__ import annotations\n"


def _future_insert_index(lines: list[str], source: str) -> int:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return 0

    if not module.body:
        return 0

    first = module.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno or 0

    return 0


def _patch_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if FUTURE_IMPORT in source:
        return False

    lines = source.splitlines(keepends=True)
    insert_at = _future_insert_index(lines, source)
    lines.insert(insert_at, FUTURE_IMPORT)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:
    spec = importlib.util.find_spec("citylearn")
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit("citylearn package not found")

    patched = []
    for location in spec.submodule_search_locations:
        package_root = Path(location)
        for path in package_root.rglob("*.py"):
            if _patch_file(path):
                patched.append(path)

    print(f"patched citylearn files for Python 3.8 annotations: {len(patched)}")


if __name__ == "__main__":
    main()

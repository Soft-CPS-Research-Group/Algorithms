from __future__ import annotations

import ast
import importlib.util
import site
from pathlib import Path


FUTURE_IMPORT = "from __future__ import annotations\n"
SKLEARN_LIBGOMP_LINK = Path("/usr/local/lib/libgomp-sklearn.so.1")


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


def _find_sklearn_libgomp() -> Path:
    candidates: list[Path] = []
    for package_root in site.getsitepackages():
        candidates.extend(Path(package_root).glob("scikit_learn.libs/libgomp*.so*"))

    if not candidates:
        raise SystemExit("scikit-learn bundled libgomp not found")

    return sorted(candidates)[0]


def _link_sklearn_libgomp() -> None:
    target = _find_sklearn_libgomp()
    if SKLEARN_LIBGOMP_LINK.exists() or SKLEARN_LIBGOMP_LINK.is_symlink():
        SKLEARN_LIBGOMP_LINK.unlink()
    SKLEARN_LIBGOMP_LINK.symlink_to(target)
    print(f"linked scikit-learn bundled libgomp: {SKLEARN_LIBGOMP_LINK} -> {target}")


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
    _link_sklearn_libgomp()


if __name__ == "__main__":
    main()

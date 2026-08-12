#!/usr/bin/env python3
"""Download only the remote simulation files needed by fixed-service oracles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


_REQUIRED_FILE = re.compile(
    r"^(?:exported_data_building_\d+(?:_battery)?_ep\d+|exported_data_pricing_ep\d+)\.csv$"
)


def _request_json(
    server: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{server.rstrip('/')}{path}",
        data=body,
        headers={} if body is None else {"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied server.
        return json.loads(response.read().decode("utf-8"))


def _request_bytes(
    server: str,
    path: str,
    *,
    payload: dict[str, Any],
    timeout: float,
) -> bytes:
    request = Request(
        f"{server.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied server.
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", default="latest")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    index = _request_json(
        args.server,
        "/simulation-data/index",
        method="POST",
        payload={"job_id": args.job_id, "session": args.session},
        timeout=args.timeout,
    )
    if not isinstance(index, dict):
        raise RuntimeError("Remote simulation-data index is not an object.")
    session = str(index.get("session") or args.session)
    files = sorted(
        str(item)
        for item in index.get("files", ())
        if isinstance(item, str) and _REQUIRED_FILE.fullmatch(item)
    )
    pricing = [name for name in files if name.startswith("exported_data_pricing_")]
    buildings = [
        name
        for name in files
        if re.fullmatch(r"exported_data_building_\d+_ep\d+\.csv", name)
    ]
    batteries = [name for name in files if "_battery_" in name]
    if len(pricing) != 1 or len(buildings) != 17 or len(batteries) != 17:
        raise RuntimeError(
            "Expected one pricing, 17 building and 17 battery files; found "
            f"{len(pricing)}, {len(buildings)} and {len(batteries)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in files:
        payload = _request_bytes(
            args.server,
            "/simulation-data/file",
            payload={
                "job_id": args.job_id,
                "session": session,
                "relative_path": relative_path,
            },
            timeout=args.timeout,
        )
        (args.output_dir / relative_path).write_bytes(payload)
        print(relative_path)

    manifest = {
        "server": args.server,
        "job_id": args.job_id,
        "session": session,
        "files": files,
    }
    (args.output_dir / "oracle_inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Embed build identity in wheels so containers/consumers do not need a Git checkout."""

import json
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithIdentity(build_py):
    def run(self):
        super().run()
        commit = os.environ.get("ALGORITHMS_COMMIT")
        dirty = None
        try:
            commit = (
                commit
                or subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            )
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        path = Path(self.build_lib) / "algorithms/_build_info.json"
        path.write_text(json.dumps({"commit": commit, "dirty": dirty}, allow_nan=False))


setup(cmdclass={"build_py": BuildWithIdentity})

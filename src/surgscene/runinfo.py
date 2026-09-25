"""Provenance for every run: git SHA, dirty flag, split hashes, config."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("git", "rev-parse", "HEAD") or None, "dirty": bool(run("git", "status", "--porcelain"))}


def split_hashes() -> dict:
    split = json.loads((ROOT / "splits/sisvse_split.json").read_text())
    return {k: v["hash"] for k, v in split.items()}


def write_runinfo(run_dir: Path, config: dict):
    info = {"config": config, "git": git_state(), "sisvse_split": split_hashes()}
    (run_dir / "runinfo.json").write_text(json.dumps(info, indent=1))

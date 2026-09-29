"""Scene semantics track: frozen GraSP split manifest and per-case label arrays (official labels).

  uv run python scripts/grasp_prepare.py

Source: GraSP 1 fps release (data/grasp/GraSP_1fps), long-term annotations of the December 2024
revision. Train = the 8 official train cases (fold1 / fold2 kept for development cross-validation),
test = the 5 official test cases, untouched until a pre-registration is committed. Frames listed in
the annotations but missing from the frame archive are dropped (one: CASE001/10972.jpg).

SurgMLLMBench's GraSP answers disagree with these labels on 18% of test and 23% of train frames
(likely built from the pre-2024 annotations); its answers are not used as ground truth here.
Writes splits/grasp_split.json (committed) and data/cache/grasp/<case>.npz (frame, phase, step).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/grasp/GraSP_1fps"
OUT = ROOT / "data/cache/grasp"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = {s: json.loads((SRC / f"annotations/grasp_long-term_{s}.json").read_text())
           for s in ("train", "test", "fold1", "fold2")}
    phases = [c["name"] for c in sorted(ann["train"]["phases_categories"], key=lambda c: c["id"])]
    steps = [c["name"] for c in sorted(ann["train"]["steps_categories"], key=lambda c: c["id"])]
    manifest = {"source": "GraSP 1 fps, long-term annotations (December 2024 revision)",
                "phases": phases, "steps": steps, "splits": {}, "dropped_missing_frames": []}
    for split in ("train", "test"):
        d = ann[split]
        by_id = {im["id"]: im for im in d["images"]}
        rows = {}
        for a in d["annotations"]:
            im = by_id[a["image_id"]]
            if not (SRC / "frames" / im["file_name"]).exists():
                manifest["dropped_missing_frames"].append(im["file_name"])
                continue
            rows.setdefault(im["video_name"], []).append((im["frame_num"], a["phases"], a["steps"]))
        cases = {}
        for case, r in sorted(rows.items()):
            r.sort()
            frame, phase, step = (np.array(x) for x in zip(*r))
            assert (np.diff(frame) == 1).all(), f"{case}: frames not contiguous"
            np.savez_compressed(OUT / f"{case}.npz", frame=frame, phase=phase, step=step)
            cases[case] = len(frame)
        names = "\n".join(f"{c}/{f:05d}" for c in sorted(rows) for f, _, _ in sorted(rows[c]))
        manifest["splits"][split] = {"cases": cases, "n_frames": sum(cases.values()),
                                     "hash": hashlib.sha256(names.encode()).hexdigest()[:16]}
    manifest["dev_folds"] = {f: sorted({im["video_name"] for im in ann[f]["images"]}) for f in ("fold1", "fold2")}
    (ROOT / "splits/grasp_split.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("phases", "steps")}, indent=1))


if __name__ == "__main__":
    main()

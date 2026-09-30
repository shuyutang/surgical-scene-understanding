"""Learned stereo (zero-shot) as a drop-in for SGM: RAFT-Stereo (v3 Phase B) and
Fast-FoundationStereo (v5 step 2).

Code and weights are fetched into third_party/ (gitignored):
  RAFT-Stereo (MIT license)
    git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo
    (cd third_party/RAFT-Stereo && bash download_models.sh)
  Fast-FoundationStereo (NVIDIA, research / non-commercial license)
    git clone https://github.com/NVlabs/Fast-FoundationStereo third_party/Fast-FoundationStereo
    research checkpoints from the Google Drive folder linked in its readme, into weights/<name>/
`disparity()` returns the same convention as proximity.disparity: left-image disparity in px,
positive, NaN where invalid. Both predict a dense field, so "invalid" means only non-positive
values. `make_stereo(name)` builds either from one string: a RAFT checkpoint ("middlebury",
"realtime@4") or "ffs:<checkpoint>@<iters>" (e.g. "ffs:23-36-37@8").
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAFT = ROOT / "third_party/RAFT-Stereo"
FFS = ROOT / "third_party/Fast-FoundationStereo"


def _use_repo(*dirs: Path):
    """Both repos import a top-level `core` package (RAFT-Stereo also bare `utils`): put this repo
    first on sys.path and drop the other's cached modules, so either model can be built after the
    other (models already built keep their classes)."""
    others = [str(d) for r in (RAFT, FFS) if r not in dirs for d in (r, r / "core")]
    sys.path[:] = [p for p in sys.path if p not in others]
    for d in reversed(dirs):
        if str(d) in sys.path:
            sys.path.remove(str(d))
        sys.path.insert(0, str(d))
    for name, mod in list(sys.modules.items()):
        top = name.split(".")[0]
        f = getattr(mod, "__file__", None) or ""
        if top in ("core", "utils", "Utils", "raft_stereo") and not any(f.startswith(str(d)) for d in dirs):
            del sys.modules[name]

CHECKPOINTS = {
    "middlebury": {"file": "raftstereo-middlebury.pth"},
    "eth3d": {"file": "raftstereo-eth3d.pth"},
    "sceneflow": {"file": "raftstereo-sceneflow.pth"},
    "realtime": {"file": "raftstereo-realtime.pth",
                 "args": {"shared_backbone": True, "n_downsample": 3, "n_gru_layers": 2, "slow_fast_gru": True}},
}


class LearnedStereo:
    def __init__(self, checkpoint: str = "middlebury", iters: int = 32, mixed_precision: bool = True):
        _use_repo(RAFT, RAFT / "core")  # the repo imports both `core.x` and bare module names
        from raft_stereo import RAFTStereo
        from utils.utils import InputPadder
        self._padder = InputPadder
        spec = CHECKPOINTS[checkpoint]
        args = dict(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4,
                    corr_radius=4, n_downsample=2, context_norm="batch", slow_fast_gru=False, n_gru_layers=3,
                    mixed_precision=mixed_precision)
        args.update(spec.get("args", {}))
        model = torch.nn.DataParallel(RAFTStereo(Namespace(**args)), device_ids=[0])
        model.load_state_dict(torch.load(RAFT / "models" / spec["file"], map_location="cuda"))
        self.model = model.module.cuda().eval()
        self.iters = iters

    @torch.no_grad()
    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """uint8 BGR rectified pair (H, W, 3) -> disparity (H, W) float32 px, NaN where <= 0."""
        to_t = lambda im: torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float()[None].cuda()
        a, b = to_t(left_bgr), to_t(right_bgr)
        padder = self._padder(a.shape, divis_by=32)
        a, b = padder.pad(a, b)
        _, flow = self.model(a, b, iters=self.iters, test_mode=True)
        d = -padder.unpad(flow)[0, 0].float().cpu().numpy()
        d[d <= 0] = np.nan
        return d


class FastFoundationStereo:
    """NVIDIA Fast-FoundationStereo (distilled FoundationStereo). The checkpoint is a pickled module."""

    def __init__(self, checkpoint: str = "23-36-37", iters: int = 8, max_disp: int = 192):
        _use_repo(FFS)
        from core.utils.utils import InputPadder
        self._padder = InputPadder
        model = torch.load(FFS / "weights" / checkpoint / "model_best_bp2_serialize.pth", map_location="cpu",
                           weights_only=False)
        model.args.valid_iters, model.args.max_disp = iters, max_disp
        self.model = model.cuda().eval()
        self.iters = iters

    @torch.no_grad()
    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """uint8 BGR rectified pair (H, W, 3) -> disparity (H, W) float32 px, NaN where <= 0."""
        to_t = lambda im: torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float()[None].cuda()
        a, b = to_t(left_bgr), to_t(right_bgr)
        padder = self._padder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            d = self.model.forward(a, b, iters=self.iters, test_mode=True, optimize_build_volume="pytorch1")
        d = padder.unpad(d.float())[0, 0].cpu().numpy()
        d[d <= 0] = np.nan
        return d


def make_stereo(name: str):
    """"middlebury" / "realtime@4" (RAFT-Stereo, 32 iterations by default) or "ffs:23-36-37@8"."""
    base, _, it = name.partition("@")
    if base.startswith("ffs:"):
        return FastFoundationStereo(base[4:], iters=int(it or 8))
    return LearnedStereo(base, iters=int(it or 32))

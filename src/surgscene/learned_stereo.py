"""v3 Phase B: learned stereo (RAFT-Stereo, zero-shot) as a drop-in for SGM.

The code and weights are fetched into third_party/RAFT-Stereo (gitignored; MIT license):
  git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo
  (cd third_party/RAFT-Stereo && bash download_models.sh)
`disparity()` returns the same convention as proximity.disparity: left-image disparity in px,
positive, NaN where invalid. RAFT-Stereo predicts a dense field, so "invalid" means only
non-positive values.
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAFT = ROOT / "third_party/RAFT-Stereo"

CHECKPOINTS = {
    "middlebury": {"file": "raftstereo-middlebury.pth"},
    "eth3d": {"file": "raftstereo-eth3d.pth"},
    "sceneflow": {"file": "raftstereo-sceneflow.pth"},
    "realtime": {"file": "raftstereo-realtime.pth",
                 "args": {"shared_backbone": True, "n_downsample": 3, "n_gru_layers": 2, "slow_fast_gru": True}},
}


class LearnedStereo:
    def __init__(self, checkpoint: str = "middlebury", iters: int = 32, mixed_precision: bool = True):
        for d in (RAFT, RAFT / "core"):  # the repo imports both `core.x` and bare module names
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
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

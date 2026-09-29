"""SurgPose stereo video reading."""

import cv2
import numpy as np

from .stage2 import RAW


def frames(traj: int, stride: int, limit: int | None = None):
    cL = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/left_video.mp4"))
    cR = cv2.VideoCapture(str(RAW / f"{traj:06d}/regular/right_video.mp4"))
    t = 0
    while True:
        okl, fl = cL.read()
        okr, fr = cR.read()
        if not (okl and okr):
            return
        if t % stride == 0:
            yield t, np.stack([fl, fr])
            if limit and t // stride + 1 >= limit:
                return
        t += 1

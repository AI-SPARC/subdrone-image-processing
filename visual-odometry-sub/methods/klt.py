import cv2
import numpy as np
from .base import BaseMethod


class KLTMethod(BaseMethod):

    def __init__(self, fb_threshold=1.0, verbose=False):
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.05,
            minDistance=15,
            blockSize=7)

        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # Erro maximo (px) no teste forward-backward.
        self.fb_threshold = fb_threshold
        self.verbose = verbose

    def get_matches(self, prev_gray, gray):

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)

        if p0 is None:
            return None, None

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
        if p1 is None:
            return None, None

        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk_params)
        if p0r is None:
            return None, None

        fb_err = np.linalg.norm(p0 - p0r, axis=2).reshape(-1)
        st = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err < self.fb_threshold)

        pts1 = p0.reshape(-1, 2)[st]
        pts2 = p1.reshape(-1, 2)[st]

        if self.verbose:
            print(f"KLT: {len(pts1)} pontos validos de {len(p0)}")

        if len(pts1) < 8:
            return None, None

        return pts1, pts2

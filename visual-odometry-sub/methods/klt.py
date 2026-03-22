import cv2
import numpy as np
from .base import BaseMethod

class KLTMethod(BaseMethod):

    def __init__(self):
        self.feature_params = dict(maxCorners=2000,
                                   qualityLevel=0.01,
                                   minDistance=7,
                                   blockSize=7)

        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def get_matches(self, prev_gray, gray):

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)

        if p0 is None:
            return None, None

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)

        if p1 is None:
            return None, None

        pts1 = p0[st == 1]
        pts2 = p1[st == 1]

        return pts1, pts2

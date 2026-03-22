"""
import cv2
import numpy as np

class VisualOdometry:

    def __init__(self, K, method="ORB"):
        self.K = K
        self.method = method

        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

        if method == "ORB":
            self.detector = cv2.ORB_create(2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        elif method == "SIFT":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        elif method == "KLT":
            self.feature_params = dict(maxCorners=2000,
                                       qualityLevel=0.01,
                                       minDistance=7,
                                       blockSize=7)
            self.lk_params = dict(winSize=(21, 21),
                                  maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS |
                                            cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def process_frame(self, prev_gray, gray):

        if self.method in ["ORB", "SIFT"]:
            kp1, des1 = self.detector.detectAndCompute(prev_gray, None)
            kp2, des2 = self.detector.detectAndCompute(gray, None)

            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        elif self.method == "KLT":
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)

            pts1 = p0[st == 1]
            pts2 = p1[st == 1]

        # Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        self.t_total += self.R_total @ t
        self.R_total = R @ self.R_total

        return self.t_total
"""

import numpy as np
import cv2

from methods.orb import ORBMethod
from methods.sift import SIFTMethod
from methods.klt import KLTMethod

class VisualOdometry:

    def __init__(self, K, method="ORB"):
        self.K = K

        if method == "ORB":
            self.method = ORBMethod()
        elif method == "SIFT":
            self.method = SIFTMethod()
        elif method == "KLT":
            self.method = KLTMethod()
        else:
            raise ValueError("Método inválido")

        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

    def process_frame(self, prev_gray, gray):

        pts1, pts2 = self.method.get_matches(prev_gray, gray)

        if pts1 is None or len(pts1) < 8:
            return self.t_total

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            cv2.RANSAC, 0.999, 1.0
        )

        if E is None:
            return self.t_total

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        self.t_total += self.R_total @ t
        self.R_total = R @ self.R_total

        return self.t_total

import cv2
import numpy as np
from .base import BaseMethod


class ORBMethod(BaseMethod):

    def __init__(self, n_features=3000, ratio=0.75):
        self.detector = cv2.ORB_create(n_features)
        # Sem crossCheck: knnMatch + ratio test filtra melhor os outliers.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio

    def get_matches(self, prev_gray, gray):
        kp1, des1 = self.detector.detectAndCompute(prev_gray, None)
        kp2, des2 = self.detector.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None

        knn = self.matcher.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return pts1, pts2

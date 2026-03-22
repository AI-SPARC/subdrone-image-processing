import cv2
import numpy as np
from .base import BaseMethod

class ORBMethod(BaseMethod):

    def __init__(self):
        self.detector = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_matches(self, prev_gray, gray):
        kp1, des1 = self.detector.detectAndCompute(prev_gray, None)
        kp2, des2 = self.detector.detectAndCompute(gray, None)

        if des1 is None or des2 is None:
            return None, None

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        return pts1, pts2

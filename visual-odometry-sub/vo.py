import numpy as np
import cv2

from methods.orb import ORBMethod
from methods.sift import SIFTMethod
from methods.klt import KLTMethod


class VisualOdometry:
    """
    Odometria visual monocular.

    IMPORTANTE (limitacao teorica): com uma unica camera e sem sensores,
    a translacao 't' devolvida por recoverPose e SEMPRE unitaria (norma 1).
    Ou seja, a escala absoluta e desconhecida e a escala relativa entre
    passos NAO e recuperada aqui. A forma da trajetoria so sera fiel se a
    velocidade do drone for aproximadamente constante entre keyframes.
    Para escala relativa correta seria preciso triangular pontos 3D e
    propagar a escala (proximo passo, se necessario).
    """

    def __init__(self, K, method="ORB", min_matches=50, min_inlier_ratio=0.3):
        self.K = K
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio

        if method == "ORB":
            self.method = ORBMethod()
        elif method == "SIFT":
            self.method = SIFTMethod()
        elif method == "KLT":
            self.method = KLTMethod()
        else:
            raise ValueError("Metodo invalido")

        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

    def process_frame(self, prev_gray, gray):
        """
        Estima o movimento entre dois frames e acumula a pose global.

        Retorna (t_total, R_total, ok):
          ok=False quando o par de frames e degenerado/ruidoso e a pose
          global e mantida inalterada (evita injetar lixo na trajetoria).
        """
        pts1, pts2 = self.method.get_matches(prev_gray, gray)

        if pts1 is None or len(pts1) < self.min_matches:
            return self.t_total, self.R_total, False

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        # E precisa ser 3x3. Quando o RANSAC devolve varios candidatos
        # empilhados (3n x 3) a cena costuma ser degenerada -> descarta.
        if E is None or E.shape != (3, 3):
            return self.t_total, self.R_total, False

        inliers = mask.ravel().astype(bool)
        if inliers.sum() < self.min_matches * self.min_inlier_ratio:
            return self.t_total, self.R_total, False

        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        n_good, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)

        # Poucos pontos com cheirality valida -> estimativa nao confiavel.
        if n_good < self.min_matches * self.min_inlier_ratio:
            return self.t_total, self.R_total, False

        # Acumulacao da pose (t unitario -> escala arbitraria).
        self.t_total = self.t_total + self.R_total @ t
        self.R_total = self.R_total @ R

        return self.t_total, self.R_total, True

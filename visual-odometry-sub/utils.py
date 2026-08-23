import cv2

def preprocess_frame(frame, target_width=960, use_clahe=True):
    """Pre-processamento do frame para ambiente subaquatico."""

    h, w = frame.shape[:2]
    scale = target_width / w
    new_dim = (target_width, int(h * scale))
    frame = cv2.resize(frame, new_dim)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    return gray

import cv2

def preprocess_frame(frame, target_width=960, use_clahe=True):
    """
    Pré-processamento automático para ambiente subaquático:
    - Redimensiona
    - Converte para grayscale
    - Aplica CLAHE
    """

    # Redimensiona mantendo proporção
    h, w = frame.shape[:2]
    scale = target_width / w
    new_dim = (target_width, int(h * scale))
    frame = cv2.resize(frame, new_dim)

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Equalização adaptativa (muito importante subaquático)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    # Leve suavização para reduzir ruído
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    return gray

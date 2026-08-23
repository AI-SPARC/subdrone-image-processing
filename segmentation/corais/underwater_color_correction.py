"""Correcao de cor para imagens subaquaticas, aplicada antes da analise de saude.

A agua absorve o vermelho primeiro, entao um coral saudavel parece descolorido.
A correcao muda os valores de saude: use o mesmo pipeline na calibracao e na
inferencia.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _to_float(img: np.ndarray) -> np.ndarray:
    """uint8 (0-255) -> float64 (0-1)."""
    return np.asarray(img, dtype=np.float64) / 255.0


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """float (0-1) -> uint8 (0-255) com clipping."""
    return np.clip(np.asarray(img) * 255.0, 0, 255).astype(np.uint8)


def estimate_cast(img_rgb: np.ndarray) -> dict:
    """Diagnostica o desvio de cor: 'azulado', 'esverdeado' ou 'neutro'."""
    f = _to_float(img_rgb)
    mr, mg, mb = float(f[..., 0].mean()), float(f[..., 1].mean()), float(f[..., 2].mean())

    dominante = max(("r", mr), ("g", mg), ("b", mb), key=lambda t: t[1])[0]
    deficit_vermelho = max(mg, mb) - mr

    if deficit_vermelho < 0.04:
        tipo = "neutro"
    elif dominante == "b":
        tipo = "azulado"
    elif dominante == "g":
        tipo = "esverdeado"
    else:
        tipo = "neutro"

    return {
        "media_r": round(mr, 4),
        "media_g": round(mg, 4),
        "media_b": round(mb, 4),
        "canal_dominante": dominante,
        "deficit_vermelho": round(deficit_vermelho, 4),
        "tipo_desvio": tipo,
    }


def white_patch_correction(
    img_rgb: np.ndarray,
    patch_bbox: tuple[int, int, int, int],
    target_gray: Optional[float] = None,
) -> np.ndarray:
    """Corrige a cor usando um trecho neutro conhecido da cena (cartao cinza).

    patch_bbox e (x1, y1, x2, y2); target_gray usa a media do patch se None.
    """
    f = _to_float(img_rgb)
    x1, y1, x2, y2 = patch_bbox
    patch = f[y1:y2, x1:x2, :]
    if patch.size == 0:
        raise ValueError("patch_bbox vazio ou fora da imagem")

    medias = patch.reshape(-1, 3).mean(axis=0)
    medias = np.maximum(medias, 1e-6)
    alvo = float(medias.mean()) if target_gray is None else float(target_gray)

    ganhos = alvo / medias
    return _to_uint8(f * ganhos[None, None, :])


def compensate_red(img_rgb: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Compensa o canal vermelho usando o verde (cenas azuladas), Ancuti et al."""
    f = _to_float(img_rgb)
    r, g = f[..., 0], f[..., 1]
    r_comp = r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g
    out = f.copy()
    out[..., 0] = np.clip(r_comp, 0.0, 1.0)
    return _to_uint8(out)


def compensate_blue(img_rgb: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Compensa o canal azul usando o verde (cenas esverdeadas)."""
    f = _to_float(img_rgb)
    b, g = f[..., 2], f[..., 1]
    b_comp = b + alpha * (g.mean() - b.mean()) * (1.0 - b) * g
    out = f.copy()
    out[..., 2] = np.clip(b_comp, 0.0, 1.0)
    return _to_uint8(out)


def shades_of_gray(img_rgb: np.ndarray, p: int = 6) -> np.ndarray:
    """Balanco de branco Shades-of-Gray (norma de Minkowski de ordem p).

    p=1 -> Gray World; p tendendo a infinito -> White Patch.
    """
    f = _to_float(img_rgb)
    norm = np.power(np.mean(np.power(f.reshape(-1, 3), p), axis=0), 1.0 / p)
    norm = np.maximum(norm, 1e-6)
    ganhos = norm.mean() / norm
    return _to_uint8(f * ganhos[None, None, :])


def gray_world(img_rgb: np.ndarray) -> np.ndarray:
    """Gray World classico: assume que a media da cena deve ser cinza."""
    return shades_of_gray(img_rgb, p=1)


def clahe_luminance(img_rgb: np.ndarray, clip_limit: float = 2.0,
                    tile: int = 8) -> np.ndarray:
    """Aumenta contraste na luminancia, preservando a cor o quanto possivel.

    Usa CLAHE do OpenCV quando disponivel; senao equaliza o histograma em NumPy.
    """
    try:
        import cv2

        bgr = img_rgb[..., ::-1]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        lab = cv2.merge((c.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[..., ::-1]
    except ImportError:
        pass

    f = _to_float(img_rgb)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    hist, bins = np.histogram((lum * 255).astype(np.uint8), bins=256, range=(0, 256))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(cdf[-1], 1.0)
    lum_eq = np.interp((lum * 255).astype(np.uint8), np.arange(256), cdf)
    escala = np.divide(lum_eq, np.maximum(lum, 1e-6))[..., None]
    return _to_uint8(np.clip(f * escala, 0.0, 1.0))


def remove_veil(img_rgb: np.ndarray, percentile: float = 1.0,
                strength: float = 1.0) -> np.ndarray:
    """Remove o veu aditivo azul-esverdeado causado pelo espalhamento.

    O veu e estimado por canal como um percentil baixo, o que exige alguma
    regiao realmente escura na cena; se o objeto de interesse for o mais escuro,
    o metodo subtrai sinal real e supersatura as cores.
    """
    f = _to_float(img_rgb)
    veu = np.percentile(f.reshape(-1, 3), percentile, axis=0) * float(strength)
    veu = np.clip(veu, 0.0, 0.95)
    out = (f - veu[None, None, :]) / np.maximum(1.0 - veu[None, None, :], 1e-6)
    return _to_uint8(np.clip(out, 0.0, 1.0))


def correct(
    img_rgb: np.ndarray,
    method: str = "auto",
    alpha: float = 1.0,
    p: int = 6,
    apply_clahe: bool = False,
    remove_backscatter: bool = False,
) -> np.ndarray:
    """Pipeline de correcao de cor subaquatica.

    method: 'auto', 'red', 'grayworld', 'shades' ou 'none'.
    Todos menos 'none' assumem media de cena acinzentada; em close-ups isso
    quebra e o balanco de branco puxa o proprio coral para o cinza.
    """
    if method == "none":
        return np.asarray(img_rgb)

    out = np.asarray(img_rgb)
    if remove_backscatter:
        out = remove_veil(out)

    if method == "auto":
        info = estimate_cast(out)
        if info["tipo_desvio"] == "azulado":
            out = compensate_red(out, alpha)
        elif info["tipo_desvio"] == "esverdeado":
            out = compensate_red(out, alpha)
            out = compensate_blue(out, alpha)
        out = shades_of_gray(out, p)
    elif method == "red":
        out = shades_of_gray(compensate_red(out, alpha), p)
    elif method == "grayworld":
        out = gray_world(out)
    elif method == "shades":
        out = shades_of_gray(out, p)
    else:
        raise ValueError(f"method desconhecido: {method}")

    if apply_clahe:
        out = clahe_luminance(out)
    return out


def simulate_underwater(img_rgb: np.ndarray, depth_factor: float = 0.6) -> np.ndarray:
    """Simula atenuacao subaquatica: derruba o vermelho, preserva azul/verde.

    depth_factor 0 = sem atenuacao, 1 = atenuacao extrema.
    """
    f = _to_float(img_rgb)
    atenuacao = np.array([1.0 - 0.85 * depth_factor,
                          1.0 - 0.25 * depth_factor,
                          1.0 - 0.05 * depth_factor])
    veu = np.array([0.02, 0.10, 0.16]) * depth_factor
    return _to_uint8(np.clip(f * atenuacao[None, None, :] + veu[None, None, :], 0, 1))


def _cena_sintetica(H: int = 160, W: int = 160, seed: int = 7):
    """Retorna (imagem, mascara_do_coral, bbox_do_cartao_cinza) de uma cena falsa."""
    rng = np.random.default_rng(seed)

    def ruido(shape):
        return rng.integers(-10, 11, shape)

    img = np.zeros((H, W, 3), dtype=np.int32)
    img[:, :] = np.array([55, 95, 115])          # agua ao fundo
    img[int(H * 0.72):, :] = np.array([195, 185, 165])  # areia

    yy, xx = np.ogrid[:H, :W]
    coral = (xx - 55) ** 2 + (yy - 80) ** 2 <= 34 ** 2
    img[coral] = np.array([120, 70, 40])         # coral marrom saudavel

    card = (slice(int(H * 0.10), int(H * 0.28)), slice(int(W * 0.70), int(W * 0.92)))
    img[card] = np.array([128, 128, 128])        # cartao cinza neutro

    img = np.clip(img + ruido((H, W, 3)), 0, 255).astype(np.uint8)
    card_bbox = (int(W * 0.70), int(H * 0.10), int(W * 0.92), int(H * 0.28))
    return img, coral, card_bbox


def _demo() -> None:
    from coral_health import analyze_mask_health

    original, coral_mask, card_bbox = _cena_sintetica()
    submerso = simulate_underwater(original, depth_factor=0.6)

    variantes = [
        ("submerso (nada feito)", submerso),
        ("grayworld", correct(submerso, "grayworld")),
        ("shades-of-gray p=6", correct(submerso, "shades")),
        ("auto (Ancuti+balanco)", correct(submerso, "auto")),
        ("cartao cinza", white_patch_correction(submerso, card_bbox)),
        ("cartao cinza + veu", white_patch_correction(remove_veil(submerso), card_bbox)),
    ]

    print("== Correcao de cor subaquatica (cena de recife sintetica) ==\n")
    print("Diagnostico do submerso:", estimate_cast(submerso), "\n")

    ref = analyze_mask_health(original, coral_mask)
    ref_rgb = np.array(ref.mean_rgb, dtype=float)
    print(f"Verdade de referencia: coral RGB={ref.mean_rgb} categoria={ref.category} "
          f"score={ref.health_score}\n")

    print(f"{'variante':<24} {'RGB do coral':<18} {'erro':>6} {'categoria':<11} "
          f"{'score':>6} {'bleach':>7}")
    print("-" * 78)
    linhas = []
    for nome, im in variantes:
        h = analyze_mask_health(im, coral_mask)
        erro = float(np.linalg.norm(np.array(h.mean_rgb, dtype=float) - ref_rgb))
        linhas.append((erro, nome))
        print(f"{nome:<24} {str(h.mean_rgb):<18} {erro:6.1f} {h.category:<11} "
              f"{h.health_score:6.1f} {h.bleaching_index:7.3f}")

    melhor = min(linhas)
    print(f"\nMenor erro de cor: '{melhor[1]}' (erro {melhor[0]:.1f})")
    print("\nErro = distancia RGB ate a cor verdadeira do coral.")
    print("Sem correcao o coral perde vermelho e e lido como palido.")
    print("O Gray World se beneficia desta cena ter media quase neutra, que e a")
    print("premissa dele; o cartao cinza nao depende da composicao da cena.")
    print("A remocao de veu supersatura porque aqui o coral e o objeto mais")
    print("escuro, e o estimador acaba subtraindo sinal real.")


if __name__ == "__main__":
    _demo()

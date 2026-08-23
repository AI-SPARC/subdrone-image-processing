"""Analise de saude de corais a partir da cor dos pixels de uma mascara.

Corais saudaveis abrigam zooxantelas, que dao pigmento. Sob estresse termico o
coral expulsa as algas e o esqueleto de carbonato fica exposto, refletindo mais
luz. A conversao RGB->HSV e feita em NumPy puro, sem depender de OpenCV nem de
PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Converte RGB (uint8 ou float 0-255) para HSV com H em [0,360) e S,V em [0,1]."""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-12
    idx = (mx == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    idx = (mx == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    idx = (mx == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

    s = np.zeros_like(mx)
    s[mx > 1e-12] = diff[mx > 1e-12] / mx[mx > 1e-12]

    return np.stack([h, s, mx], axis=-1)


# Aproximacao dos 6 niveis da Coral Health Chart do CoralWatch.
CORALWATCH_REFERENCE = {
    1: (245, 240, 235),
    2: (225, 205, 180),
    3: (205, 170, 130),
    4: (180, 130, 85),
    5: (150, 95, 55),
    6: (110, 65, 35),
}


def nearest_coralwatch_level(mean_rgb: np.ndarray) -> int:
    """Retorna o nivel CoralWatch (1-6) mais proximo da cor media."""
    ref = np.array(list(CORALWATCH_REFERENCE.values()), dtype=np.float64)
    levels = np.array(list(CORALWATCH_REFERENCE.keys()))
    dist = np.linalg.norm(ref - np.asarray(mean_rgb, dtype=np.float64), axis=1)
    return int(levels[int(np.argmin(dist))])


@dataclass
class HealthResult:
    category: str
    health_score: float
    bleaching_index: float
    coralwatch_level: int
    mean_rgb: tuple
    mean_hsv: tuple
    n_pixels: int
    extra: dict = field(default_factory=dict)


# A chave "regra" escolhe o sinal que domina a decisao. Em imagem subaquatica
# crua a saturacao nao separa coral vivo de branqueado (d de Cohen 0.23),
# enquanto o brilho separa (d = 2.46), por isso o padrao e 'brilho'. Use
# calibrate_health.py para reajustar em outro material.
DEFAULT_THRESHOLDS = {
    "regra": "brilho",

    "sat_healthy": 0.35,
    "sat_pale": 0.18,
    "val_bright": 0.55,

    "bri_bleached": 0.71,
    # Faixa intermediaria aberta de proposito: o Coralscapes nao tem classe
    # 'palido', e otimizar so para duas classes colapsaria bri_pale em 0.70.
    "bri_pale": 0.62,

    "idx_bleached": 0.50,
    "idx_pale": 0.42,

    "val_dark": 0.22,
}


def bleaching_index(s: float, v: float) -> float:
    """Indice de branqueamento V*(1-S): 1 para branco puro, 0 no escuro."""
    return float(np.clip(v * (1.0 - s), 0.0, 1.0))


def classificar(s: float, v: float, thresholds: Optional[dict] = None) -> str:
    """Classifica a saude a partir da saturacao e do brilho medios.

    Usada tanto por analyze_mask_health quanto por calibrate_health.py, para
    que calibracao e inferencia nao divirjam.
    """
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    if v < t["val_dark"]:
        return "escuro"

    regra = t.get("regra", "saturacao")

    if regra == "brilho":
        if v >= t["bri_bleached"]:
            return "branqueado"
        if v >= t["bri_pale"]:
            return "palido"
        return "saudavel"

    if regra == "indice":
        idx = bleaching_index(s, v)
        if idx >= t["idx_bleached"]:
            return "branqueado"
        if idx >= t["idx_pale"]:
            return "palido"
        return "saudavel"

    if s >= t["sat_healthy"]:
        return "saudavel"
    if s >= t["sat_pale"]:
        return "palido"
    return "branqueado" if v >= t["val_bright"] else "palido"


def analyze_mask_health(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    thresholds: Optional[dict] = None,
    ignore_dark: bool = True,
) -> HealthResult:
    """Analisa a saude de um coral dada a imagem RGB e sua mascara booleana."""
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    mask_bool = np.asarray(mask_bool, dtype=bool)
    pixels = image_rgb[mask_bool]
    if pixels.size == 0:
        return HealthResult(
            category="indefinido",
            health_score=0.0,
            bleaching_index=0.0,
            coralwatch_level=0,
            mean_rgb=(0, 0, 0),
            mean_hsv=(0, 0, 0),
            n_pixels=0,
            extra={"aviso": "mascara vazia"},
        )

    hsv = rgb_to_hsv(pixels)
    v_all = hsv[:, 2]
    dark_frac = float(np.mean(v_all < t["val_dark"]))

    if ignore_dark:
        keep = v_all >= t["val_dark"]
        if np.count_nonzero(keep) >= max(10, int(0.05 * len(v_all))):
            pixels_used = pixels[keep]
            hsv_used = hsv[keep]
        else:
            pixels_used = pixels
            hsv_used = hsv
    else:
        pixels_used = pixels
        hsv_used = hsv

    mean_rgb = pixels_used.mean(axis=0)
    mean_h = float(np.mean(hsv_used[:, 0]))
    mean_s = float(np.mean(hsv_used[:, 1]))
    mean_v = float(np.mean(hsv_used[:, 2]))

    idx = bleaching_index(mean_s, mean_v)

    if t.get("regra", "saturacao") == "saturacao":
        health_score = float(np.clip(mean_s * 100.0 * 1.15 - (idx * 15.0), 0.0, 100.0))
    else:
        health_score = float(np.clip((1.0 - idx) * 100.0, 0.0, 100.0))

    return HealthResult(
        category=classificar(mean_s, mean_v, t),
        health_score=round(health_score, 1),
        bleaching_index=round(idx, 3),
        coralwatch_level=nearest_coralwatch_level(mean_rgb),
        mean_rgb=tuple(int(round(c)) for c in mean_rgb),
        mean_hsv=(round(mean_h, 1), round(mean_s, 3), round(mean_v, 3)),
        n_pixels=int(pixels.shape[0]),
        extra={"fracao_pixels_escuros": round(dark_frac, 3)},
    )


def aplicar_classe_do_modelo(hr: "HealthResult", nome_classe: str) -> "HealthResult":
    """Deixa o modelo decidir vivo/morto e a cor graduar a saude do que vive.

    A cor classifica 72% do coral morto como saudavel, porque esqueleto coberto
    de alga reflete de forma parecida com tecido pigmentado. Modifica e devolve
    o proprio hr; nomes de classe desconhecidos passam sem alteracao.
    """
    hr.extra["classe_modelo"] = nome_classe

    if nome_classe == "coral_morto":
        hr.extra["categoria_por_cor"] = hr.category
        hr.category = "morto"
        hr.health_score = 0.0
    elif nome_classe == "coral_branqueado" and hr.category == "saudavel":
        hr.extra["conflito_modelo_cor"] = True

    return hr


def summarize_reef(results: list[HealthResult]) -> dict:
    """Agrega os corais de uma imagem em um resumo do recife."""
    valid = [r for r in results if r.n_pixels > 0]
    if not valid:
        return {"n_corais": 0}

    total_px = sum(r.n_pixels for r in valid)
    bleached_px = sum(r.n_pixels for r in valid if r.category == "branqueado")
    pale_px = sum(r.n_pixels for r in valid if r.category == "palido")
    dead_px = sum(r.n_pixels for r in valid if r.category == "morto")

    by_cat: dict[str, int] = {}
    for r in valid:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1

    # Coral morto entra na area, que interessa para cobertura, mas fica fora
    # das medias de saude.
    vivos = [r for r in valid if r.category != "morto"]
    resumo = {
        "n_corais": len(valid),
        "saude_media": (
            round(float(np.mean([r.health_score for r in vivos])), 1)
            if vivos else None
        ),
        "indice_branqueamento_medio": (
            round(float(np.mean([r.bleaching_index for r in vivos])), 3)
            if vivos else None
        ),
        "area_branqueada_pct": round(100.0 * bleached_px / total_px, 1),
        "area_palida_pct": round(100.0 * pale_px / total_px, 1),
        "contagem_por_categoria": by_cat,
    }
    if dead_px:
        resumo["area_morta_pct"] = round(100.0 * dead_px / total_px, 1)
    return resumo


def _demo() -> None:
    """Gera manchas sinteticas de coral e imprime a analise."""
    rng = np.random.default_rng(42)
    H = W = 200
    img = np.zeros((H, W, 3), dtype=np.uint8)

    def patch(color, cx, cy, r, noise=15):
        yy, xx = np.ogrid[:H, :W]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        n = rng.integers(-noise, noise + 1, size=(H, W, 3))
        base = np.clip(np.array(color)[None, None, :] + n, 0, 255).astype(np.uint8)
        img[m] = base[m]
        return m

    amostras = [
        ("saudavel", patch((120, 70, 40), 50, 50, 30)),
        ("palido", patch((200, 175, 150), 150, 50, 30)),
        ("branqueado", patch((245, 243, 240), 100, 150, 30)),
    ]

    print("Analise de amostras sinteticas\n")
    print(f"{'amostra':>11}  {'S':>6} {'V':>6}   {'regra=brilho':<14} "
          f"{'regra=saturacao':<15}")
    print("-" * 62)
    for nome, m in amostras:
        r_bri = analyze_mask_health(img, m, thresholds={"regra": "brilho"})
        r_sat = analyze_mask_health(img, m, thresholds={"regra": "saturacao"})
        s, v = r_bri.mean_hsv[1], r_bri.mean_hsv[2]
        print(f"{nome:>11}  {s:6.3f} {v:6.3f}   {r_bri.category:<14} "
              f"{r_sat.category:<15}")

    print("\nAs cores sinteticas simulam material fora d'agua, onde a saturacao")
    print("ainda discrimina. A regra ativa foi calibrada em imagem subaquatica")
    print("crua, e por isso classifica a amostra clara como branqueada.\n")

    results = []
    for nome, m in amostras:
        r = analyze_mask_health(img, m)
        results.append(r)
        print(f"[{nome:>10}] categoria={r.category:<11} score={r.health_score:5.1f} "
              f"bleach={r.bleaching_index:.3f} CW={r.coralwatch_level} "
              f"RGB={r.mean_rgb} HSV={r.mean_hsv}")

    print("\nResumo do recife:")
    for k, v in summarize_reef(results).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()

"""Inferencia de segmentacao de corais com analise de saude por cor.

Roda um modelo YOLO-seg na imagem/pasta/video e salva, por imagem, um overlay
por categoria de saude e um JSON com o resultado.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from coral_health import (HealthResult, analyze_mask_health,
                          aplicar_classe_do_modelo, summarize_reef)
from underwater_color_correction import correct as corrigir_cor
from underwater_color_correction import estimate_cast, white_patch_correction


CATEGORY_COLOR_BGR = {
    "saudavel": (60, 180, 75),
    "palido": (0, 200, 255),
    "branqueado": (60, 60, 230),
    "morto": (80, 80, 80),
    "escuro": (150, 150, 150),
    "indefinido": (200, 200, 200),
}


def _load_cv2():
    try:
        import cv2

        return cv2
    except ImportError as e:
        raise SystemExit(
            "OpenCV nao encontrado. Instale com: pip install opencv-python"
        ) from e


def _erodir(mask_bool: np.ndarray, px: int, cv2) -> np.ndarray:
    """Erode a mascara `px` pixels, mas nunca ate ela desaparecer."""
    if px <= 0:
        return mask_bool
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    erodida = cv2.erode(mask_bool.astype(np.uint8), k, iterations=1).astype(bool)
    if erodida.sum() < max(20, 0.15 * mask_bool.sum()):
        return mask_bool
    return erodida


def run(
    model_path: str,
    source: str,
    out_dir: str,
    conf: float = 0.25,
    color_correct: str = "none",
    erode_px: int = 0,
    gray_card: tuple[int, int, int, int] | None = None,
    save_corrected: bool = False,
) -> None:
    cv2 = _load_cv2()
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, verbose=False)

    for ri, res in enumerate(results):
        img_bgr = res.orig_img
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        diagnostico = estimate_cast(img_rgb)

        # A analise de cor usa a imagem CORRIGIDA; a visualizacao usa a original.
        if gray_card is not None:
            img_analise = white_patch_correction(img_rgb, gray_card)
            metodo_usado = "cartao_cinza"
        elif color_correct != "none":
            img_analise = corrigir_cor(img_rgb, method=color_correct)
            metodo_usado = color_correct
        else:
            img_analise = img_rgb
            metodo_usado = "nenhum"

        if save_corrected and metodo_usado != "nenhum":
            cv2.imwrite(os.path.join(out_dir, f"coral_{ri:03d}_corrigida.jpg"),
                        cv2.cvtColor(img_analise, cv2.COLOR_RGB2BGR))

        overlay = img_bgr.copy()
        health_results: list[HealthResult] = []

        if res.masks is None:
            print(f"[{ri}] Nenhum coral detectado.")
        else:
            masks = res.masks.data.cpu().numpy()
            H, W = img_rgb.shape[:2]
            for i in range(masks.shape[0]):
                m = masks[i]
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = m > 0.5
                mask_medicao = _erodir(mask_bool, erode_px, cv2)

                hr = analyze_mask_health(img_analise, mask_medicao)
                # Vivo x morto vem do modelo (textura), nao da cor.
                try:
                    nome_classe = res.names[int(res.boxes.cls[i].item())]
                except Exception:
                    nome_classe = ""
                aplicar_classe_do_modelo(hr, nome_classe)
                try:
                    hr.extra["conf_deteccao"] = round(
                        float(res.boxes.conf[i].item()), 3
                    )
                except Exception:
                    pass
                health_results.append(hr)

                color = CATEGORY_COLOR_BGR.get(hr.category, (200, 200, 200))
                overlay[mask_bool] = color
                ys, xs = np.where(mask_bool)
                if len(xs):
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(
                        img_bgr, f"{hr.category} {hr.health_score:.0f}",
                        (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2, cv2.LINE_AA,
                    )

        blended = cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)

        base = f"coral_{ri:03d}"
        out_img = os.path.join(out_dir, base + "_saude.jpg")
        out_json = os.path.join(out_dir, base + "_saude.json")
        cv2.imwrite(out_img, blended)

        report = {
            "processamento": {
                "correcao_de_cor": metodo_usado,
                "erode_px": erode_px,
                "conf": conf,
                "diagnostico_cor_original": diagnostico,
            },
            "corais": [hr.__dict__ for hr in health_results],
            "resumo": summarize_reef(health_results),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{ri}] {len(health_results)} corais -> {out_img}")
        print(f"     desvio de cor original: {diagnostico['tipo_desvio']} "
              f"| correcao: {metodo_usado} | erode: {erode_px}px")
        print(f"     resumo: {report['resumo']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segmentacao de corais + analise de saude por cor"
    )
    ap.add_argument("--model", required=True, help="caminho do best.pt (YOLO-seg)")
    ap.add_argument("--source", required=True, help="imagem, pasta ou video")
    ap.add_argument("--out", default="saida_corais", help="pasta de saida")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--color-correct", default="none",
                    choices=["none", "auto", "red", "grayworld", "shades"],
                    help="correcao de cor subaquatica antes de medir a saude")
    ap.add_argument("--erode-px", type=int, default=0,
                    help="erode a mascara N px antes de medir a cor "
                         "(evita vazamento para o fundo; sugestao: 2 a 4)")
    ap.add_argument("--gray-card", default=None,
                    help="x1,y1,x2,y2 de um cartao cinza na cena "
                         "(tem prioridade sobre --color-correct)")
    ap.add_argument("--save-corrected", action="store_true",
                    help="salva tambem a imagem corrigida, para conferir")
    a = ap.parse_args()

    card = None
    if a.gray_card:
        partes = [int(v) for v in a.gray_card.split(",")]
        if len(partes) != 4:
            ap.error("--gray-card espera 4 numeros: x1,y1,x2,y2")
        card = tuple(partes)

    run(a.model, a.source, a.out, a.conf, a.color_correct, a.erode_px,
        card, a.save_corrected)


if __name__ == "__main__":
    main()

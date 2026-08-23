"""Camada de inferencia unificada: roda na mesma imagem a deteccao de
peixe-leao e a segmentacao de corais com analise de saude, e devolve um
relatorio combinado (JSON + imagem anotada).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reutiliza o modulo de saude que vive em corais/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "corais"))
from coral_health import (HealthResult, analyze_mask_health,  # noqa: E402
                          aplicar_classe_do_modelo, summarize_reef)
from underwater_color_correction import correct as corrigir_cor  # noqa: E402
from underwater_color_correction import estimate_cast  # noqa: E402


CATEGORY_COLOR_BGR = {
    "saudavel": (60, 180, 75),
    "palido": (0, 200, 255),
    "branqueado": (60, 60, 230),
    "morto": (80, 80, 80),
    "escuro": (150, 150, 150),
    "indefinido": (200, 200, 200),
}


def _erodir(mask_bool, px, cv2):
    """Erode a mascara px pixels (evita medir cor do fundo), sem apaga-la."""
    if px <= 0:
        return mask_bool
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    e = cv2.erode(mask_bool.astype(np.uint8), k, iterations=1).astype(bool)
    return mask_bool if e.sum() < max(20, 0.15 * mask_bool.sum()) else e


def run(lionfish_model, coral_model, source, out_dir, conf=0.25,
        color_correct="none", erode_px=0):
    import cv2
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    det = YOLO(lionfish_model) if lionfish_model else None
    seg = YOLO(coral_model) if coral_model else None

    seg_results = seg.predict(source=source, conf=conf, verbose=False) if seg else []
    det_results = det.predict(source=source, conf=conf, verbose=False) if det else []
    n = max(len(seg_results), len(det_results))

    for i in range(n):
        seg_res = seg_results[i] if i < len(seg_results) else None
        det_res = det_results[i] if i < len(det_results) else None

        base_res = seg_res or det_res
        img_bgr = base_res.orig_img.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        overlay = img_bgr.copy()

        diagnostico = estimate_cast(img_rgb)
        img_analise = (img_rgb if color_correct == "none"
                       else corrigir_cor(img_rgb, method=color_correct))

        health_results: list[HealthResult] = []
        if seg_res is not None and seg_res.masks is not None:
            masks = seg_res.masks.data.cpu().numpy()
            for j in range(masks.shape[0]):
                m = masks[j]
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = m > 0.5
                hr = analyze_mask_health(img_analise, _erodir(mask_bool, erode_px, cv2))
                # Vivo x morto vem do modelo (textura), nao da cor.
                try:
                    nome_classe = seg_res.names[int(seg_res.boxes.cls[j].item())]
                except Exception:
                    nome_classe = ""
                aplicar_classe_do_modelo(hr, nome_classe)
                health_results.append(hr)
                overlay[mask_bool] = CATEGORY_COLOR_BGR.get(hr.category, (200, 200, 200))

        img_bgr = cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)

        lionfish = []
        if det_res is not None and det_res.boxes is not None:
            boxes = det_res.boxes.xyxy.cpu().numpy()
            confs = det_res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(boxes, confs):
                lionfish.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                 "conf": float(c)})
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 0, 255), 2)
                cv2.putText(img_bgr, f"peixe-leao {c:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        base = f"frame_{i:03d}"
        out_img = os.path.join(out_dir, base + ".jpg")
        out_json = os.path.join(out_dir, base + ".json")
        cv2.imwrite(out_img, img_bgr)

        report = {
            "processamento": {
                "correcao_de_cor": color_correct,
                "erode_px": erode_px,
                "conf": conf,
                "diagnostico_cor_original": diagnostico,
            },
            "peixe_leao": {"n_deteccoes": len(lionfish), "deteccoes": lionfish},
            "corais": {
                "instancias": [hr.__dict__ for hr in health_results],
                "resumo": summarize_reef(health_results),
            },
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{i}] peixe-leao={len(lionfish)} corais={len(health_results)} -> {out_img}")


def main():
    ap = argparse.ArgumentParser(description="Pipeline unificado peixe-leao + coral")
    ap.add_argument("--lionfish-model", default=None, help="best.pt de deteccao (peixe-leao)")
    ap.add_argument("--coral-model", default=None, help="best.pt de segmentacao (coral)")
    ap.add_argument("--source", required=True, help="imagem, pasta ou video")
    ap.add_argument("--out", default="saida_unificada")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--color-correct", default="none",
                    choices=["none", "auto", "red", "grayworld", "shades"],
                    help="correcao de cor subaquatica antes de medir a saude")
    ap.add_argument("--erode-px", type=int, default=0,
                    help="erode a mascara N px antes de medir a cor (sugestao: 2 a 4)")
    args = ap.parse_args()
    if not args.lionfish_model and not args.coral_model:
        ap.error("informe ao menos um modelo (--lionfish-model e/ou --coral-model)")
    run(args.lionfish_model, args.coral_model, args.source, args.out, args.conf,
        args.color_correct, args.erode_px)


if __name__ == "__main__":
    main()

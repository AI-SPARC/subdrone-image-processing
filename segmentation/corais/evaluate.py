"""Avaliacao de treino e inferencia, com interpretacao das metricas.

Subcomandos: val, curves, health, glossario, selftest.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


GLOSSARIO = """
(B) refere-se a caixa e (M) a mascara. Para analise de cor o grupo relevante e
(M), ja que a saude e medida nos pixels da mascara.

  P          do que foi apontado como coral, quanto era coral
  R          dos corais reais, quantos foram encontrados
  mAP50      acerto medio considerando IoU >= 0.50
  mAP50-95   media do mAP para IoU de 0.50 a 0.95; penaliza contorno impreciso
  IoU        intersecao sobre uniao entre mascara predita e verdadeira

Contorno impreciso deixa pixels de fundo entrarem na media de cor. Erodir a
mascara antes de medir e preferivel a arriscar vazamento para a areia clara.
"""


def cmd_val(model_path: str, data_yaml: str, imgsz: int = 640,
            split: str = "test") -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Instale o ultralytics: pip install ultralytics")

    model = YOLO(model_path)
    r = model.val(data=data_yaml, imgsz=imgsz, split=split, plots=True, verbose=True)

    print("RESULTADOS")
    print("-" * 40)

    def mostra(nome, obj):
        try:
            print(f"{nome:<12} P={obj.mp:.3f}  R={obj.mr:.3f}  "
                  f"mAP50={obj.map50:.3f}  mAP50-95={obj.map:.3f}")
        except Exception:
            print(f"{nome:<12} (indisponivel)")

    caixa = getattr(r, "box", None)
    mascara = getattr(r, "seg", None) or getattr(r, "mask", None)
    if caixa is not None:
        mostra("Box (B)", caixa)
    if mascara is not None:
        mostra("Mask (M)", mascara)

        m50, m5095 = mascara.map50, mascara.map
        print()
        if m50 < 0.40:
            print("mAP50(M) baixo: deteccao pouco confiavel, faltam dados ou")
            print("os rotulos precisam de revisao.")
        elif m5095 < 0.35:
            print("Deteccao razoavel mas contorno impreciso: risco de fundo")
            print("contaminar a media de cor. Use erode_px na inferencia.")
        else:
            print("Contorno adequado para analise de cor.")
        if mascara.mr < 0.5:
            print("Recall baixo: cobertura do recife sera subestimada.")
        if mascara.mp < 0.5:
            print("Precision baixa: muitos falsos positivos.")


def _achar_coluna(cabecalho: list[str], *pistas: str) -> int | None:
    """Acha a primeira coluna cujo nome contem todas as pistas."""
    for i, nome in enumerate(cabecalho):
        n = nome.strip().lower()
        if all(p.lower() in n for p in pistas):
            return i
    return None


def cmd_curves(csv_path: str, plot: bool = False) -> dict:
    """Le results.csv do Ultralytics, resume a evolucao e diagnostica."""
    with open(csv_path, "r", encoding="utf-8") as f:
        linhas = list(csv.reader(f))
    if len(linhas) < 2:
        raise SystemExit("results.csv sem dados")

    cab = [c.strip() for c in linhas[0]]
    dados = []
    for l in linhas[1:]:
        if not l or not l[0].strip():
            continue
        dados.append([float(v) if v.strip() not in ("", "nan") else np.nan
                      for v in l])
    arr = np.array(dados, dtype=float)

    # Prefere metricas de mascara (M); cai para caixa (B) se nao houver.
    i_map = (_achar_coluna(cab, "map50-95", "(m)") or
             _achar_coluna(cab, "map50-95", "(b)") or
             _achar_coluna(cab, "map50-95"))
    i_map50 = (_achar_coluna(cab, "map50", "(m)") or
               _achar_coluna(cab, "map50", "(b)") or
               _achar_coluna(cab, "map50"))
    i_epoca = _achar_coluna(cab, "epoch") or 0

    cols_train = [i for i, c in enumerate(cab) if c.lower().startswith("train/")]
    cols_val = [i for i, c in enumerate(cab) if c.lower().startswith("val/")]

    print(f"Arquivo: {csv_path}")
    print(f"Epocas registradas: {len(arr)}\n")

    if i_map is not None:
        serie = arr[:, i_map]
        melhor = int(np.nanargmax(serie))
        print(f"Melhor epoca por mAP50-95: {int(arr[melhor, i_epoca])} "
              f"(mAP50-95={serie[melhor]:.4f}"
              + (f", mAP50={arr[melhor, i_map50]:.4f}" if i_map50 else "") + ")")
        print(f"Ultima epoca: mAP50-95={serie[-1]:.4f}")

        n = len(serie)
        if n >= 9:
            t1 = np.nanmax(serie[: n // 3])
            t3 = np.nanmax(serie[2 * n // 3:])
            if t3 - t1 < 0.01:
                print("\n - Curva praticamente ESTAGNADA: o modelo parou de aprender.")
                print("   Provavel limite de dados/rotulos, nao de epocas.")
        if melhor < 0.6 * n:
            print(f"\n - O melhor resultado veio na epoca {int(arr[melhor, i_epoca])} "
                  f"de {n}: treinar mais tempo nao ajudou (early stopping ok).")

    if cols_train and cols_val and len(arr) >= 6:
        tr = np.nansum(arr[:, cols_train], axis=1)
        va = np.nansum(arr[:, cols_val], axis=1)
        meio = len(arr) // 2
        d_tr = np.nanmean(tr[meio:]) - np.nanmean(tr[:meio])
        d_va = np.nanmean(va[meio:]) - np.nanmean(va[:meio])
        print(f"\nPerda total treino: {tr[0]:.3f} -> {tr[-1]:.3f}")
        print(f"Perda total valid.: {va[0]:.3f} -> {va[-1]:.3f}")
        if d_tr < 0 and d_va > 0:
            print(" - OVERFITTING: treino melhora e validacao piora. Reduza epocas,")
            print("   aumente augmentation/weight_decay ou colete mais imagens.")
        elif d_tr > 0:
            print(" - Perda de treino NAO caiu: learning rate ou dados suspeitos.")
        else:
            print(" - Sem sinal claro de overfitting.")

    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            if i_map is not None:
                ax[0].plot(arr[:, i_epoca], arr[:, i_map], label="mAP50-95")
            if i_map50 is not None:
                ax[0].plot(arr[:, i_epoca], arr[:, i_map50], label="mAP50")
            ax[0].set_xlabel("epoca"); ax[0].set_ylabel("metrica")
            ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("Metricas")

            if cols_train:
                ax[1].plot(arr[:, i_epoca], np.nansum(arr[:, cols_train], axis=1),
                           label="perda treino")
            if cols_val:
                ax[1].plot(arr[:, i_epoca], np.nansum(arr[:, cols_val], axis=1),
                           label="perda validacao")
            ax[1].set_xlabel("epoca"); ax[1].legend(); ax[1].grid(alpha=0.3)
            ax[1].set_title("Perdas")

            saida = os.path.join(os.path.dirname(csv_path) or ".", "curvas.png")
            fig.tight_layout(); fig.savefig(saida, dpi=120)
            print(f"\nGrafico salvo em {saida}")
        except ImportError:
            print("\n(matplotlib nao instalado; use pip install matplotlib "
                  "para gerar o grafico)")

    return {"epocas": len(arr)}


def cmd_health(dir_json: str, out_csv: str | None = None) -> None:
    """Agrega os relatorios de saude gerados pelo predict_coral.py."""
    arquivos = sorted(glob.glob(os.path.join(dir_json, "*.json")))
    if not arquivos:
        raise SystemExit(f"nenhum .json em {dir_json}")

    todas = []
    for caminho in arquivos:
        with open(caminho, "r", encoding="utf-8") as f:
            rel = json.load(f)
        # Aceita o formato do predict_coral.py e do pipeline_unificado.py
        instancias = rel.get("corais")
        if isinstance(instancias, dict):
            instancias = instancias.get("instancias", [])
        for inst in instancias or []:
            inst["_arquivo"] = os.path.basename(caminho)
            todas.append(inst)

    if not todas:
        raise SystemExit("nenhuma instancia de coral encontrada nos JSONs")

    scores = np.array([i.get("health_score", 0.0) for i in todas], dtype=float)
    bleach = np.array([i.get("bleaching_index", 0.0) for i in todas], dtype=float)
    pixels = np.array([i.get("n_pixels", 0) for i in todas], dtype=float)
    cats: dict[str, int] = {}
    for i in todas:
        c = i.get("category", "?")
        cats[c] = cats.get(c, 0) + 1

    print(f"Imagens analisadas: {len(arquivos)}")
    print(f"Instancias de coral: {len(todas)}\n")
    print(f"health_score:    media={scores.mean():.1f} mediana="
          f"{np.median(scores):.1f} min={scores.min():.1f} max={scores.max():.1f}")
    print(f"bleaching_index: media={bleach.mean():.3f} mediana="
          f"{np.median(bleach):.3f}\n")

    print("Distribuicao por categoria (por instancia e por AREA):")
    total_px = max(pixels.sum(), 1.0)
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        area = sum(i.get("n_pixels", 0) for i in todas if i.get("category") == c)
        print(f"  {c:<12} {n:>5} inst ({100*n/len(todas):5.1f}%)   "
              f"area {100*area/total_px:5.1f}%")

    print("\nA porcentagem por area e a mais relevante para cobertura.")

    if out_csv:
        campos = ["_arquivo", "category", "health_score", "bleaching_index",
                  "coralwatch_level", "n_pixels", "mean_rgb", "mean_hsv"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(campos)
            for i in todas:
                w.writerow([i.get(c, "") for c in campos])
        print(f"\nCSV salvo em {out_csv}")


def selftest() -> None:
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="eval_selftest_")
    try:
        print("== Autoteste de evaluate ==\n")

        # results.csv sintetico com overfitting proposital.
        cab = ["epoch", "train/box_loss", "train/seg_loss",
               "metrics/mAP50(M)", "metrics/mAP50-95(M)",
               "val/box_loss", "val/seg_loss"]
        linhas = [cab]
        for e in range(1, 21):
            tr = max(0.10, 1.20 - 0.05 * e)
            va = 0.90 if e <= 10 else 0.90 + 0.03 * (e - 10)
            m50 = min(0.62, 0.10 + 0.055 * e)
            m5095 = min(0.41, 0.05 + 0.036 * e)
            linhas.append([e, round(tr, 4), round(tr * 0.8, 4),
                           round(m50, 4), round(m5095, 4),
                           round(va, 4), round(va * 0.8, 4)])
        csv_path = os.path.join(tmp, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(linhas)

        print("-- curves (espera detectar overfitting) --")
        cmd_curves(csv_path, plot=False)

        print("\n-- health --")
        for k in range(2):
            rel = {
                "corais": [
                    {"category": "saudavel", "health_score": 70.0,
                     "bleaching_index": 0.15, "coralwatch_level": 5,
                     "n_pixels": 5000, "mean_rgb": [120, 70, 40],
                     "mean_hsv": [22.0, 0.66, 0.47]},
                    {"category": "branqueado", "health_score": 5.0,
                     "bleaching_index": 0.90, "coralwatch_level": 1,
                     "n_pixels": 15000, "mean_rgb": [240, 238, 235],
                     "mean_hsv": [30.0, 0.02, 0.94]},
                ],
                "resumo": {"n_corais": 2},
            }
            with open(os.path.join(tmp, f"coral_{k:03d}_saude.json"), "w",
                      encoding="utf-8") as f:
                json.dump(rel, f)

        out_csv = os.path.join(tmp, "resumo.csv")
        cmd_health(tmp, out_csv)
        assert os.path.exists(out_csv), "CSV de resumo nao foi criado"
        with open(out_csv, encoding="utf-8") as f:
            n_linhas = len(f.read().strip().splitlines())
        assert n_linhas == 5, f"esperava 1 cabecalho + 4 instancias, veio {n_linhas}"
        print("\nOK: agregacao de saude gerou 4 instancias no CSV")
        print("\n== Autoteste concluido com sucesso ==")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliacao e interpretacao de resultados")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("val", help="validacao oficial + interpretacao")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])

    p = sub.add_parser("curves", help="analisa results.csv de um treino")
    p.add_argument("--csv", required=True)
    p.add_argument("--plot", action="store_true")

    p = sub.add_parser("health", help="agrega JSONs de saude")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None)

    sub.add_parser("glossario", help="apenas imprime o glossario de metricas")
    sub.add_parser("selftest", help="autoteste com dados sinteticos")

    a = ap.parse_args()
    if a.cmd == "val":
        cmd_val(a.model, a.data, a.imgsz, a.split)
    elif a.cmd == "curves":
        cmd_curves(a.csv, a.plot)
    elif a.cmd == "health":
        cmd_health(a.dir, a.out)
    elif a.cmd == "glossario":
        print(GLOSSARIO)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()

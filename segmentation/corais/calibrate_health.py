"""Calibra os limiares de `coral_health` a partir de recortes ja rotulados.

Espera <dir>/saudavel, <dir>/palido e <dir>/branqueado com imagens dentro,
geradas com o mesmo pipeline de correcao de cor usado na inferencia.
Subcomandos: report, fit, selftest.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from coral_health import DEFAULT_THRESHOLDS, bleaching_index, classificar, rgb_to_hsv

CATEGORIAS = ("saudavel", "palido", "branqueado")

REGRAS = ("saturacao", "brilho", "indice")


def _ler_imagem_rgb(caminho: str) -> np.ndarray | None:
    try:
        import cv2

        bgr = cv2.imread(caminho, cv2.IMREAD_COLOR)
        return None if bgr is None else bgr[..., ::-1]
    except ImportError:
        pass
    try:
        from PIL import Image

        return np.array(Image.open(caminho).convert("RGB"))
    except ImportError:
        raise SystemExit(
            "Preciso de OpenCV ou Pillow para ler imagens.\n"
            "  pip install opencv-python   (ou)   pip install pillow"
        )


def carregar_amostras(raiz: str) -> list[tuple[str, float, float]]:
    """Le a pasta de calibracao e devolve [(categoria, saturacao, valor), ...].

    As medias ignoram pixels escuros, como em coral_health.analyze_mask_health.
    """
    amostras: list[tuple[str, float, float]] = []
    for cat in CATEGORIAS:
        pasta = os.path.join(raiz, cat)
        if not os.path.isdir(pasta):
            print(f"  aviso: pasta ausente: {pasta}")
            continue
        arquivos = [f for f in sorted(os.listdir(pasta))
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif"))]
        for nome in arquivos:
            img = _ler_imagem_rgb(os.path.join(pasta, nome))
            if img is None:
                continue
            hsv = rgb_to_hsv(img.reshape(-1, 3))
            v = hsv[:, 2]
            manter = v >= DEFAULT_THRESHOLDS["val_dark"]
            if np.count_nonzero(manter) < max(10, int(0.05 * len(v))):
                manter = np.ones_like(v, dtype=bool)
            amostras.append((cat, float(hsv[manter, 1].mean()),
                             float(hsv[manter, 2].mean())))
        print(f"  {cat}: {len(arquivos)} imagens")
    return amostras


def acuracia(amostras, t: dict) -> float:
    if not amostras:
        return 0.0
    acertos = sum(1 for cat, s, v in amostras if classificar(s, v, t) == cat)
    return acertos / len(amostras)


def matriz_confusao(amostras, t: dict) -> dict:
    rotulos = list(CATEGORIAS) + ["escuro"]
    m = {r: {p: 0 for p in rotulos} for r in rotulos}
    for cat, s, v in amostras:
        m.setdefault(cat, {p: 0 for p in rotulos})
        m[cat][classificar(s, v, t)] += 1
    return m


def report(amostras) -> None:
    print("\n== Distribuicao de saturacao (S) e brilho (V) por categoria ==\n")
    print(f"{'categoria':<12} {'n':>4} {'S media':>9} {'S p10':>7} {'S p90':>7} "
          f"{'V media':>9}")
    print("-" * 56)
    for cat in CATEGORIAS:
        vals = [(s, v) for c, s, v in amostras if c == cat]
        if not vals:
            print(f"{cat:<12} {'0':>4}  (sem amostras)")
            continue
        s_arr = np.array([s for s, _ in vals])
        v_arr = np.array([v for _, v in vals])
        print(f"{cat:<12} {len(vals):>4} {s_arr.mean():9.3f} "
              f"{np.percentile(s_arr, 10):7.3f} {np.percentile(s_arr, 90):7.3f} "
              f"{v_arr.mean():9.3f}")

    print("\nFaixas de S sobrepostas entre 'saudavel' e 'branqueado' indicam")
    print("que a saturacao sozinha nao separa as classes nestes dados.")

    separabilidade(amostras)

    print(f"\nAcuracia com os limiares atuais: "
          f"{acuracia(amostras, DEFAULT_THRESHOLDS)*100:.1f}%")


def separabilidade(amostras) -> dict[str, float]:
    """Usa o d de Cohen para ver qual sinal separa saudavel de branqueado."""
    vivos = [(s, v) for c, s, v in amostras if c == "saudavel"]
    branq = [(s, v) for c, s, v in amostras if c == "branqueado"]
    if not vivos or not branq:
        return {}

    sinais = {
        "saturacao (S)": (lambda s, v: s),
        "brilho (V)": (lambda s, v: v),
        "indice V*(1-S)": bleaching_index,
    }

    print("\n== Qual sinal separa 'saudavel' de 'branqueado'? ==\n")
    print(f"{'sinal':<18} {'saudavel':>10} {'branqueado':>12} {'d de Cohen':>12}")
    print("-" * 56)

    resultado: dict[str, float] = {}
    for nome, f in sinais.items():
        a = np.array([f(s, v) for s, v in vivos])
        b = np.array([f(s, v) for s, v in branq])
        dp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        d = abs(b.mean() - a.mean()) / dp if dp > 0 else 0.0
        resultado[nome] = float(d)
        print(f"{nome:<18} {a.mean():>10.3f} {b.mean():>12.3f} {d:>12.2f}")

    melhor = max(resultado, key=resultado.get)
    print(f"\n|d| < 0.5 separa mal, ~0.8 e moderado, > 1.2 e bom.")
    print(f"Melhor sinal nestes dados: {melhor} (d={resultado[melhor]:.2f})")
    return resultado


def _grade_da_regra(regra: str):
    """Gera os dicionarios de limiares a testar para uma familia de regra."""
    if regra == "saturacao":
        for sh in np.round(np.arange(0.20, 0.66, 0.01), 3):
            for sp in np.round(np.arange(0.05, 0.46, 0.01), 3):
                if sp >= sh:
                    continue
                for vb in np.round(np.arange(0.35, 0.81, 0.05), 3):
                    yield {"regra": "saturacao", "sat_healthy": float(sh),
                           "sat_pale": float(sp), "val_bright": float(vb)}
    elif regra == "brilho":
        for bb in np.round(np.arange(0.30, 0.96, 0.01), 3):
            for bp in np.round(np.arange(0.20, 0.91, 0.01), 3):
                if bp >= bb:
                    continue
                yield {"regra": "brilho", "bri_bleached": float(bb),
                       "bri_pale": float(bp)}
    else:  # indice
        for ib in np.round(np.arange(0.20, 0.96, 0.01), 3):
            for ip in np.round(np.arange(0.10, 0.91, 0.01), 3):
                if ip >= ib:
                    continue
                yield {"regra": "indice", "idx_bleached": float(ib),
                       "idx_pale": float(ip)}


def fit(amostras, salvar: str | None = "limiares_calibrados.json",
        regras: tuple[str, ...] = REGRAS) -> dict:
    """Busca em grade os limiares que maximizam a acuracia.

    Testa cada familia de regra e fica com a melhor, porque trocar o sinal de
    decisao costuma importar mais do que ajustar o limiar de saturacao.
    """
    if not amostras:
        raise SystemExit("sem amostras para calibrar")

    print("\n== Busca em grade por familia de regra ==\n")
    melhor = (-1.0, dict(DEFAULT_THRESHOLDS))
    for regra in regras:
        melhor_regra = (-1.0, dict(DEFAULT_THRESHOLDS))
        for ajuste in _grade_da_regra(regra):
            t = dict(DEFAULT_THRESHOLDS)
            t.update(ajuste)
            a = acuracia(amostras, t)
            if a > melhor_regra[0]:
                melhor_regra = (a, t)
        print(f"  regra '{regra}':  melhor acuracia = {melhor_regra[0]*100:.1f}%")
        if melhor_regra[0] > melhor[0]:
            melhor = melhor_regra

    acc, t = melhor
    base = acuracia(amostras, DEFAULT_THRESHOLDS)
    print(f"\nRegra vencedora: '{t.get('regra')}'")
    print(f"Acuracia antes (limiares padrao): {base*100:.1f}%")
    print(f"Acuracia depois (calibrado):      {acc*100:.1f}%")

    print("\nMatriz de confusao (linha = verdade, coluna = predito):")
    m = matriz_confusao(amostras, t)
    cols = list(CATEGORIAS) + ["escuro"]
    print("            " + "".join(f"{c:>12}" for c in cols))
    for r in cols:
        if r not in m:
            continue
        print(f"{r:<12}" + "".join(f"{m[r][c]:>12}" for c in cols))

    print("\nCole isto em coral_health.py (DEFAULT_THRESHOLDS):\n")
    print("DEFAULT_THRESHOLDS = {")
    for k, v in t.items():
        print(f'    "{k}": {json.dumps(v)},')
    print("}")

    if salvar:
        with open(salvar, "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2)
        print(f"\nSalvo em {salvar} (carregue com json.load e passe em")
        print("analyze_mask_health(..., thresholds=...))")

    if acc < 0.7:
        print("\nAcuracia baixa. Verifique se os recortes tem fundo contaminando")
        print("a cor, se a correcao de cor esta consistente e se os rotulos sao")
        print("confiaveis.")
    return t


def selftest() -> None:
    """Gera amostras sinteticas separaveis e verifica que a calibracao melhora."""
    rng = np.random.default_rng(0)
    amostras: list[tuple[str, float, float]] = []

    # Limiares padrao ficam deslocados de proposito nestes dados.
    for _ in range(60):
        amostras.append(("saudavel", float(np.clip(rng.normal(0.55, 0.05), 0, 1)),
                         float(np.clip(rng.normal(0.45, 0.05), 0, 1))))
    for _ in range(60):
        amostras.append(("palido", float(np.clip(rng.normal(0.30, 0.04), 0, 1)),
                         float(np.clip(rng.normal(0.70, 0.05), 0, 1))))
    for _ in range(60):
        amostras.append(("branqueado", float(np.clip(rng.normal(0.10, 0.03), 0, 1)),
                         float(np.clip(rng.normal(0.90, 0.04), 0, 1))))

    print("== Autoteste de calibrate_health (dados sinteticos) ==")
    report(amostras)
    base = acuracia(amostras, DEFAULT_THRESHOLDS)
    t = fit(amostras, salvar=None)
    final = acuracia(amostras, t)

    assert final >= base, "calibracao nao deveria piorar a acuracia"
    assert final > 0.9, f"esperava acuracia alta em dados separaveis, veio {final}"
    print(f"\nOK: acuracia {base*100:.1f}% -> {final*100:.1f}%")
    print("== Autoteste concluido com sucesso ==")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibra limiares de saude de coral")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report", help="distribuicoes de S/V por categoria")
    p.add_argument("--dir", required=True)

    p = sub.add_parser("fit", help="busca em grade dos melhores limiares")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default="limiares_calibrados.json")

    sub.add_parser("selftest", help="autoteste com dados sinteticos")

    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return

    print(f"Lendo amostras de {a.dir} ...")
    amostras = carregar_amostras(a.dir)
    if not amostras:
        raise SystemExit(
            "Nenhuma amostra encontrada. Esperado: <dir>/saudavel, <dir>/palido, "
            "<dir>/branqueado com imagens dentro."
        )
    if a.cmd == "report":
        report(amostras)
    else:
        fit(amostras, a.out)


if __name__ == "__main__":
    main()

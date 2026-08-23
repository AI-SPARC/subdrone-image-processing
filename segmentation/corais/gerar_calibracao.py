"""Gera o conjunto de calibracao do `calibrate_health.py` recortando
instancias de coral das mascaras do Coralscapes, agrupadas por categoria de
saude, com o fundo em preto.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Mesmos IDs do map_coral.json, agrupados por categoria de saude.
IDS_POR_CATEGORIA = {
    "saudavel": [6, 17, 21, 22, 25, 27, 28, 31, 34, 36],
    "branqueado": [4, 16, 19, 33],
    "morto": [3, 20, 23, 32, 37],
}


def gerar(raiz: Path, split: str, saida: Path, min_area: int,
          max_por_classe: int | None, padding: int,
          color_correct: str | None = None) -> dict:
    import cv2

    dir_img = raiz / split / "images"
    dir_msk = raiz / split / "masks"
    if not dir_msk.is_dir():
        raise SystemExit(
            f"pasta nao encontrada: {dir_msk}\n"
            "Rode antes: python exportar_coralscapes.py"
        )

    id_para_cat = {i: cat for cat, ids in IDS_POR_CATEGORIA.items() for i in ids}
    contagem = {cat: 0 for cat in IDS_POR_CATEGORIA}
    for cat in IDS_POR_CATEGORIA:
        (saida / cat).mkdir(parents=True, exist_ok=True)

    mascaras = sorted(dir_msk.glob("*.png"))
    if not mascaras:
        raise SystemExit(f"nenhuma mascara em {dir_msk}")

    print(f"Processando {len(mascaras)} mascaras de '{split}'...")

    for n_arq, caminho_msk in enumerate(mascaras, 1):
        if max_por_classe and all(c >= max_por_classe for c in contagem.values()):
            print("  cota atingida em todas as categorias; parando cedo")
            break

        caminho_img = dir_img / (caminho_msk.stem + ".jpg")
        if not caminho_img.exists():
            continue

        m = cv2.imread(str(caminho_msk), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(caminho_img), cv2.IMREAD_COLOR)
        if m is None or img is None:
            continue
        if color_correct:
            # Corrige a imagem inteira: o balanco de branco precisa da cena toda.
            from underwater_color_correction import correct as corrigir_cor
            img = corrigir_cor(img[..., ::-1], method=color_correct)[..., ::-1]
        if m.ndim == 3:
            m = m[..., 0]
        H, W = m.shape[:2]

        for valor in np.unique(m):
            cat = id_para_cat.get(int(valor))
            if cat is None:
                continue
            if max_por_classe and contagem[cat] >= max_por_classe:
                continue

            binaria = (m == valor).astype(np.uint8)
            n_comp, rotulos, stats, _ = cv2.connectedComponentsWithStats(
                binaria, connectivity=8
            )
            # rotulo 0 e o fundo
            for c in range(1, n_comp):
                if max_por_classe and contagem[cat] >= max_por_classe:
                    break
                area = stats[c, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                x = stats[c, cv2.CC_STAT_LEFT]
                y = stats[c, cv2.CC_STAT_TOP]
                w = stats[c, cv2.CC_STAT_WIDTH]
                h = stats[c, cv2.CC_STAT_HEIGHT]
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(W, x + w + padding), min(H, y + h + padding)

                recorte = img[y1:y2, x1:x2].copy()
                dentro = (rotulos[y1:y2, x1:x2] == c)
                # Fundo em preto: o calibrate_health descarta pixels escuros.
                recorte[~dentro] = 0

                nome = f"{caminho_msk.stem}_id{int(valor)}_{c:03d}.png"
                cv2.imwrite(str(saida / cat / nome), recorte)
                contagem[cat] += 1

        if n_arq % 50 == 0:
            print(f"  {n_arq}/{len(mascaras)} mascaras | {contagem}")

    print(f"\nRecortes gerados em {saida.resolve()}:")
    for cat, n in contagem.items():
        print(f"  {cat:<12} {n}")
    return contagem


def diagnostico(saida: Path) -> None:
    """Mede como a analise de cor classifica cada categoria verdadeira."""
    import cv2

    from coral_health import analyze_mask_health

    print("\n== Diagnostico: o que a analise de COR responde para cada "
          "categoria verdadeira ==\n")

    categorias = list(IDS_POR_CATEGORIA)
    for cat in categorias:
        pasta = saida / cat
        arquivos = sorted(pasta.glob("*.png")) if pasta.is_dir() else []
        if not arquivos:
            print(f"{cat:<12} (sem recortes)")
            continue

        preditos: dict[str, int] = {}
        scores = []
        for caminho in arquivos:
            bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = bgr[..., ::-1]
            mask = np.any(bgr > 0, axis=2)  # fundo preto fora do coral
            if mask.sum() < 20:
                continue
            r = analyze_mask_health(rgb, mask)
            preditos[r.category] = preditos.get(r.category, 0) + 1
            scores.append(r.health_score)

        total = sum(preditos.values()) or 1
        resumo = "  ".join(
            f"{k}={v} ({100*v/total:.0f}%)"
            for k, v in sorted(preditos.items(), key=lambda kv: -kv[1])
        )
        print(f"{cat:<12} n={total:<5} score medio={np.mean(scores):5.1f}  ->  {resumo}")

    print("\nCoral morto tende a ser lido como saudavel pela cor, ja que")
    print("esqueleto com alga reflete de forma parecida com tecido vivo. Essa")
    print("distincao depende da classe do modelo, nao da cor.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gera o conjunto de calibracao a partir do Coralscapes"
    )
    ap.add_argument("--raiz", default=".", help="pasta com train/valid/test")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--saida", default="calib")
    ap.add_argument("--min-area", type=int, default=3000,
                    help="area minima da instancia em px (recortes pequenos "
                         "dao media de cor instavel)")
    ap.add_argument("--max-por-classe", type=int, default=300)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--diagnostico", action="store_true",
                    help="apenas analisa os recortes ja gerados")
    ap.add_argument("--color-correct", default=None,
                    choices=["auto", "red", "grayworld", "shades"],
                    help="aplica correcao de cor antes de recortar. Permite "
                         "testar se corrigir a cor faz a saturacao voltar a "
                         "separar coral vivo de branqueado.")
    a = ap.parse_args()

    saida = Path(a.saida)
    if not a.diagnostico:
        gerar(Path(a.raiz), a.split, saida, a.min_area, a.max_por_classe,
              a.padding, a.color_correct)
    diagnostico(saida)

    print("\nProximo passo:")
    print("  python calibrate_health.py report --dir calib")
    print("  python calibrate_health.py fit --dir calib")


if __name__ == "__main__":
    main()

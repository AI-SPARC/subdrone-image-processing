"""Baixa o dataset Coralscapes do Hugging Face e exporta para disco na
estrutura de pastas usada pelo projeto (train/valid/test com images/ e
masks/).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

DATASET = "EPFL-ECEO/coralscapes"

# split no HF -> nome da pasta local (o data.yaml usa 'valid', nao 'validation')
SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def verificar_amostra(amostra) -> None:
    """Sanity check na primeira amostra, antes de gastar tempo com tudo."""
    img = amostra["image"]
    lbl = amostra["label"]
    arr = np.array(lbl)

    print("  imagem      :", img.size, img.mode)
    print("  mascara     :", lbl.size, lbl.mode, "->", arr.shape, arr.dtype)
    valores = np.unique(arr)
    print("  IDs na 1a mascara:", valores.tolist())

    if arr.ndim != 2:
        raise SystemExit(
            f"ERRO: mascara com {arr.ndim} dimensoes (esperado 2). "
            "O formato do dataset mudou; revise este script."
        )
    if valores.max() > 39:
        raise SystemExit(
            f"ERRO: valor {valores.max()} acima de 39 na mascara. "
            "Provavelmente a paleta foi aplicada e os IDs se perderam."
        )
    print("  OK: IDs dentro do intervalo esperado (0..39)\n")


def exportar(destino: Path, splits: list[str], limit: int | None,
             max_width: int | None, sobrescrever: bool) -> None:
    from datasets import load_dataset

    total_geral = 0
    primeiro = True
    for split_hf in splits:
        split_local = SPLITS[split_hf]
        dir_img = destino / split_local / "images"
        dir_msk = destino / split_local / "masks"
        dir_img.mkdir(parents=True, exist_ok=True)
        dir_msk.mkdir(parents=True, exist_ok=True)

        # Um split por vez: o Hugging Face baixa so os shards daquele split.
        print(f"Carregando split '{split_hf}' de {DATASET} ...")
        t0 = time.time()
        parte = load_dataset(DATASET, split=split_hf)
        print(f"  {len(parte)} imagens disponiveis ({time.time()-t0:.0f}s)")

        if primeiro:
            print("\nVerificando a primeira amostra:")
            verificar_amostra(parte[0])
            primeiro = False

        n = len(parte) if limit is None else min(limit, len(parte))
        print(f"[{split_local}] exportando {n} imagens -> {dir_img.parent}")

        t0 = time.time()
        escritos = pulados = 0
        for i in range(n):
            nome = f"{split_local}_{i:05d}"
            caminho_img = dir_img / f"{nome}.jpg"
            caminho_msk = dir_msk / f"{nome}.png"

            if not sobrescrever and caminho_img.exists() and caminho_msk.exists():
                pulados += 1
                continue

            amostra = parte[i]
            img = amostra["image"].convert("RGB")
            arr = np.array(amostra["label"]).astype(np.uint8)

            if max_width and img.width > max_width:
                escala = max_width / img.width
                novo = (max_width, int(round(img.height * escala)))
                img = img.resize(novo, Image.BILINEAR)
                # NEAREST na mascara: interpolar inventaria IDs de classe.
                arr = np.array(
                    Image.fromarray(arr, mode="L").resize(novo, Image.NEAREST)
                )

            img.save(caminho_img, quality=95)
            Image.fromarray(arr, mode="L").save(caminho_msk, optimize=True)
            escritos += 1

            if escritos and escritos % 100 == 0:
                dt = time.time() - t0
                resta = (n - i - 1) * dt / max(escritos, 1)
                print(f"    {i+1}/{n}  ({dt:.0f}s decorridos, "
                      f"~{resta/60:.1f} min restantes)")

        print(f"[{split_local}] pronto: {escritos} escritos, {pulados} ja existiam "
              f"({time.time()-t0:.0f}s)\n")
        total_geral += escritos + pulados

    print(f"Total: {total_geral} imagens em {destino.resolve()}")
    print("\nProximo passo:")
    print("  python convert_annotations.py mask2yolo --masks train\\masks "
          "--out train\\labels --class-map map_coral.json --min-area 1500")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baixa e exporta o Coralscapes")
    ap.add_argument("--destino", default=".",
                    help="pasta raiz (padrao: diretorio atual)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS),
                    choices=list(SPLITS),
                    help="quais splits exportar")
    ap.add_argument("--limit", type=int, default=None,
                    help="exporta so as N primeiras imagens de cada split (teste)")
    ap.add_argument("--max-width", type=int, default=1024,
                    help="reduz a largura das imagens (padrao 1024). Use 0 "
                         "para manter a resolucao original.")
    ap.add_argument("--sobrescrever", action="store_true",
                    help="reescreve arquivos que ja existem")
    a = ap.parse_args()

    exportar(Path(a.destino), a.splits, a.limit, a.max_width or None,
             a.sobrescrever)


if __name__ == "__main__":
    main()

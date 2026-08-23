"""Conversores e validadores de anotacoes para o formato YOLO de segmentacao.

Formato de destino: <id_classe> x1 y1 ... xn yn, normalizado, minimo 3 pontos.
Subcomandos: coco2yolo, mask2yolo, bbox2poly, validate, stats, selftest.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def coco2yolo(json_path: str, out_dir: str, class_offset: int = 0) -> None:
    """Converte um COCO JSON com poligonos em arquivos YOLO-seg.

    Anotacoes em RLE ou com iscrowd=1 sao ignoradas; use mask2yolo nesses casos.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    imagens = {im["id"]: im for im in coco.get("images", [])}
    categorias = coco.get("categories", [])
    # YOLO exige ids de classe contiguos comecando em 0.
    cat_ids = [c["id"] for c in categorias]
    cat_para_idx = {cid: i + class_offset for i, cid in enumerate(cat_ids)}
    nomes = [c.get("name", str(c["id"])) for c in categorias]

    por_imagem: dict[int, list[str]] = defaultdict(list)
    ignoradas = 0
    convertidas = 0

    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or isinstance(seg, dict) or ann.get("iscrowd", 0) == 1:
            ignoradas += 1
            continue

        img = imagens.get(ann["image_id"])
        if img is None:
            ignoradas += 1
            continue
        W, H = float(img["width"]), float(img["height"])
        cls = cat_para_idx.get(ann["category_id"])
        if cls is None:
            ignoradas += 1
            continue

        for poly in seg:
            if len(poly) < 6:  # menos de 3 pontos
                ignoradas += 1
                continue
            coords = []
            for i in range(0, len(poly) - 1, 2):
                x = min(max(poly[i] / W, 0.0), 1.0)
                y = min(max(poly[i + 1] / H, 0.0), 1.0)
                coords += [f"{x:.6f}", f"{y:.6f}"]
            por_imagem[ann["image_id"]].append(f"{cls} " + " ".join(coords))
            convertidas += 1

    # Arquivos vazios sao escritos de proposito: servem como negativos.
    for img_id, img in imagens.items():
        nome = os.path.splitext(os.path.basename(img["file_name"]))[0]
        with open(os.path.join(out_dir, nome + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(por_imagem.get(img_id, [])))
            if por_imagem.get(img_id):
                f.write("\n")

    print(f"[coco2yolo] {convertidas} poligonos escritos em {out_dir}")
    print(f"[coco2yolo] {ignoradas} anotacoes ignoradas (RLE/crowd/invalidas)")
    print(f"[coco2yolo] nc: {len(nomes)}")
    print(f"[coco2yolo] names: {nomes}")


def mask2yolo(
    masks_dir: str,
    out_dir: str,
    class_map_path: str | None = None,
    min_area: int = 80,
    epsilon_frac: float = 0.002,
    ignore_values: tuple[int, ...] = (0,),
) -> None:
    """Converte mascaras PNG indexadas (1 canal, valor = classe) em YOLO-seg.

    class_map_path: JSON {"valor_no_png": id_classe_yolo}; sem ele cada valor
    distinto recebe um id sequencial. Como a mascara e semantica, corais que se
    tocam viram uma unica instancia.
    """
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "mask2yolo precisa de OpenCV. Instale com: pip install opencv-python"
        )

    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    mapa: dict[int, int] | None = None
    if class_map_path:
        with open(class_map_path, "r", encoding="utf-8") as f:
            bruto = json.load(f)
        # Chaves iniciadas por "_" sao comentarios dentro do JSON.
        mapa = {int(k): int(v) for k, v in bruto.items()
                if not str(k).startswith("_")}
        print(f"[mask2yolo] mapa carregado: {len(mapa)} valores -> "
              f"{sorted(set(mapa.values()))}")

    auto_mapa: dict[int, int] = {}
    arquivos = [f for f in sorted(os.listdir(masks_dir))
                if f.lower().endswith((".png", ".bmp", ".tif", ".tiff"))]
    if not arquivos:
        raise SystemExit(f"nenhuma mascara encontrada em {masks_dir}")

    total_poly = 0
    for nome_arq in arquivos:
        m = cv2.imread(os.path.join(masks_dir, nome_arq), cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f"  aviso: nao consegui ler {nome_arq}")
            continue
        if m.ndim == 3:
            m = m[..., 0]
        H, W = m.shape[:2]

        linhas = []
        for valor in np.unique(m):
            valor = int(valor)
            if valor in ignore_values:
                continue
            if mapa is not None:
                if valor not in mapa:
                    continue
                cls = mapa[valor]
            else:
                cls = auto_mapa.setdefault(valor, len(auto_mapa))

            binaria = (m == valor).astype(np.uint8)
            contornos, _ = cv2.findContours(
                binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contornos:
                if cv2.contourArea(c) < min_area:
                    continue
                eps = epsilon_frac * cv2.arcLength(c, True)
                aprox = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
                if len(aprox) < 3:
                    continue
                coords = []
                for x, y in aprox:
                    coords += [f"{min(max(x / W, 0.0), 1.0):.6f}",
                               f"{min(max(y / H, 0.0), 1.0):.6f}"]
                linhas.append(f"{cls} " + " ".join(coords))
                total_poly += 1

        base = os.path.splitext(nome_arq)[0]
        with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))
            if linhas:
                f.write("\n")

    print(f"[mask2yolo] {len(arquivos)} mascaras -> {total_poly} poligonos em {out_dir}")
    if mapa is None:
        print(f"[mask2yolo] mapeamento automatico valor_png -> classe: {auto_mapa}")
        print("[mask2yolo] use --class-map para controlar isso explicitamente")


def bbox2poly(labels_dir: str, out_dir: str) -> None:
    """Converte rotulos YOLO de caixa (`cls cx cy w h`) em poligonos de 4 pontos.

    O retangulo inclui fundo e contamina a media de cor: serve so como rascunho.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_arq = n_box = 0
    for nome in sorted(os.listdir(labels_dir)):
        if not nome.endswith(".txt"):
            continue
        linhas_out = []
        with open(os.path.join(labels_dir, nome), "r", encoding="utf-8") as f:
            for linha in f:
                p = linha.split()
                if len(p) != 5:
                    continue
                cls, cx, cy, w, h = p[0], *map(float, p[1:])
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                coords = []
                for x, y in pts:
                    coords += [f"{min(max(x, 0.0), 1.0):.6f}",
                               f"{min(max(y, 0.0), 1.0):.6f}"]
                linhas_out.append(f"{cls} " + " ".join(coords))
                n_box += 1
        with open(os.path.join(out_dir, nome), "w", encoding="utf-8") as f:
            f.write("\n".join(linhas_out))
            if linhas_out:
                f.write("\n")
        n_arq += 1
    print(f"[bbox2poly] {n_arq} arquivos, {n_box} caixas -> poligonos em {out_dir}")
    print("[bbox2poly] AVISO: retangulos incluem fundo; refine antes de medir cor.")


def validate(labels_dir: str, nc: int | None = None) -> bool:
    """Valida arquivos YOLO-seg. Retorna True se nao houver erros."""
    erros: list[str] = []
    avisos: list[str] = []
    n_arq = n_inst = n_vazios = 0

    arquivos = [f for f in sorted(os.listdir(labels_dir)) if f.endswith(".txt")]
    if not arquivos:
        print(f"[validate] ERRO: nenhum .txt em {labels_dir}")
        return False

    for nome in arquivos:
        n_arq += 1
        caminho = os.path.join(labels_dir, nome)
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = [l for l in f.read().splitlines() if l.strip()]
        if not linhas:
            n_vazios += 1
            continue

        for i, linha in enumerate(linhas, 1):
            p = linha.split()
            local = f"{nome}:{i}"
            if len(p) == 5:
                erros.append(f"{local}: parece CAIXA (cls cx cy w h) e nao poligono; "
                             f"use bbox2poly ou re-anote como segmentacao")
                continue
            if len(p) < 7:
                erros.append(f"{local}: poucos valores ({len(p)}); "
                             f"minimo e classe + 3 pontos = 7")
                continue
            try:
                cls = int(p[0])
                coords = [float(v) for v in p[1:]]
            except ValueError:
                erros.append(f"{local}: valor nao numerico")
                continue

            if len(coords) % 2 != 0:
                erros.append(f"{local}: numero impar de coordenadas ({len(coords)})")
                continue
            if cls < 0 or (nc is not None and cls >= nc):
                erros.append(f"{local}: id de classe {cls} fora de [0,{nc})")
            fora = [c for c in coords if c < -1e-6 or c > 1 + 1e-6]
            if fora:
                erros.append(f"{local}: {len(fora)} coordenadas fora de [0,1] "
                             f"(esqueceu de normalizar?)")
            if len(coords) // 2 < 4:
                avisos.append(f"{local}: poligono com apenas {len(coords)//2} pontos")
            n_inst += 1

    print(f"[validate] {n_arq} arquivos, {n_inst} instancias, {n_vazios} vazios "
          f"(negativos)")
    for a in avisos[:15]:
        print(f"  aviso: {a}")
    if len(avisos) > 15:
        print(f"  ... e mais {len(avisos)-15} avisos")
    for e in erros[:25]:
        print(f"  ERRO: {e}")
    if len(erros) > 25:
        print(f"  ... e mais {len(erros)-25} erros")

    ok = not erros
    print(f"[validate] resultado: {'OK' if ok else 'FALHOU'}")
    return ok


def stats(labels_dir: str) -> None:
    """Resume o dataset: instancias por classe e complexidade dos poligonos."""
    por_classe: dict[int, int] = defaultdict(int)
    pontos: list[int] = []
    areas: list[float] = []
    n_arq = n_vazios = 0

    for nome in sorted(os.listdir(labels_dir)):
        if not nome.endswith(".txt"):
            continue
        n_arq += 1
        with open(os.path.join(labels_dir, nome), "r", encoding="utf-8") as f:
            linhas = [l for l in f.read().splitlines() if l.strip()]
        if not linhas:
            n_vazios += 1
        for linha in linhas:
            p = linha.split()
            if len(p) < 7:
                continue
            por_classe[int(p[0])] += 1
            c = [float(v) for v in p[1:]]
            xs, ys = c[0::2], c[1::2]
            pontos.append(len(xs))
            # Area do poligono pela formula do shoelace, em fracao da imagem.
            a = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                a += xs[i] * ys[j] - xs[j] * ys[i]
            areas.append(abs(a) / 2.0)

    print(f"[stats] arquivos: {n_arq} (vazios/negativos: {n_vazios})")
    print(f"[stats] instancias totais: {sum(por_classe.values())}")
    for cls in sorted(por_classe):
        print(f"[stats]   classe {cls}: {por_classe[cls]}")
    if pontos:
        pontos_ord = sorted(pontos)
        areas_ord = sorted(areas)
        med = pontos_ord[len(pontos_ord) // 2]
        med_a = areas_ord[len(areas_ord) // 2]
        print(f"[stats] pontos por poligono: min={min(pontos)} mediana={med} "
              f"max={max(pontos)}")
        print(f"[stats] area relativa: mediana={med_a:.4f} "
              f"min={min(areas):.5f} max={max(areas):.4f}")
        peq = sum(1 for a in areas if a < 0.001)
        if peq:
            print(f"[stats] aviso: {peq} instancias com area < 0.1% da imagem "
                  f"(muito pequenas; considere aumentar imgsz ou recortar)")


def selftest() -> None:
    """Testa coco2yolo, bbox2poly, validate e stats em dados sinteticos."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="conv_selftest_")
    try:
        print("== Autoteste de convert_annotations ==\n")

        coco = {
            "images": [{"id": 1, "file_name": "recife_01.jpg",
                        "width": 200, "height": 100}],
            "categories": [{"id": 7, "name": "coral"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "iscrowd": 0,
                 "segmentation": [[10, 10, 50, 10, 50, 40, 10, 40]]},
                {"id": 2, "image_id": 1, "category_id": 7, "iscrowd": 0,
                 "segmentation": [[100, 20, 160, 20, 130, 80]]},
                # Esta deve ser ignorada (RLE)
                {"id": 3, "image_id": 1, "category_id": 7, "iscrowd": 1,
                 "segmentation": {"counts": "abc", "size": [100, 200]}},
            ],
        }
        json_path = os.path.join(tmp, "ann.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)

        out_coco = os.path.join(tmp, "labels_coco")
        coco2yolo(json_path, out_coco)

        with open(os.path.join(out_coco, "recife_01.txt"), encoding="utf-8") as f:
            conteudo = f.read().strip().splitlines()
        assert len(conteudo) == 2, f"esperava 2 linhas, veio {len(conteudo)}"
        # Primeiro ponto: x=10/200=0.05, y=10/100=0.10
        primeiro = conteudo[0].split()
        assert primeiro[0] == "0", "classe deveria ser remapeada para 0"
        assert abs(float(primeiro[1]) - 0.05) < 1e-6
        assert abs(float(primeiro[2]) - 0.10) < 1e-6
        print("  OK: coco2yolo normalizou e remapeou a classe corretamente\n")

        print("-- validate no resultado do coco2yolo --")
        assert validate(out_coco, nc=1) is True
        print()
        print("-- stats no resultado do coco2yolo --")
        stats(out_coco)
        print()

        dir_box = os.path.join(tmp, "labels_box")
        os.makedirs(dir_box)
        with open(os.path.join(dir_box, "img1.txt"), "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.4\n")
        out_poly = os.path.join(tmp, "labels_poly")
        print("-- bbox2poly --")
        bbox2poly(dir_box, out_poly)
        with open(os.path.join(out_poly, "img1.txt"), encoding="utf-8") as f:
            linha = f.read().split()
        assert len(linha) == 9, f"esperava 1 classe + 8 coords, veio {len(linha)}"
        assert abs(float(linha[1]) - 0.4) < 1e-6, "x1 deveria ser 0.5-0.1=0.4"
        assert abs(float(linha[2]) - 0.3) < 1e-6, "y1 deveria ser 0.5-0.2=0.3"
        print("  OK: caixa convertida em retangulo de 4 pontos\n")

        print("-- validate detectando problemas de proposito --")
        dir_ruim = os.path.join(tmp, "labels_ruins")
        os.makedirs(dir_ruim)
        with open(os.path.join(dir_ruim, "ruim.txt"), "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.4\n")             # caixa, nao poligono
            f.write("0 1.5 0.1 0.2 0.2 0.3 0.3\n")     # coord > 1
            f.write("9 0.1 0.1 0.2 0.2 0.3 0.3\n")     # classe fora de nc=1
        ok = validate(dir_ruim, nc=1)
        assert ok is False, "validate deveria ter falhado"
        print("  OK: validate apontou caixa, coordenada fora de [0,1] e classe invalida\n")

        print("== Autoteste concluido com sucesso ==")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Conversores/validadores de anotacao para YOLO-seg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("coco2yolo", help="COCO JSON com poligonos -> YOLO-seg")
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("mask2yolo", help="mascaras PNG indexadas -> YOLO-seg")
    p.add_argument("--masks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--class-map", default=None,
                   help='JSON {"valor_png": id_classe}')
    p.add_argument("--min-area", type=int, default=80)
    p.add_argument("--epsilon-frac", type=float, default=0.002)

    p = sub.add_parser("bbox2poly", help="caixas YOLO -> poligonos retangulares")
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("validate", help="valida arquivos YOLO-seg")
    p.add_argument("--labels", required=True)
    p.add_argument("--nc", type=int, default=None)

    p = sub.add_parser("stats", help="estatisticas do dataset")
    p.add_argument("--labels", required=True)

    sub.add_parser("selftest", help="autoteste com dados sinteticos")

    a = ap.parse_args()
    if a.cmd == "coco2yolo":
        coco2yolo(a.json, a.out)
    elif a.cmd == "mask2yolo":
        mask2yolo(a.masks, a.out, a.class_map, a.min_area, a.epsilon_frac)
    elif a.cmd == "bbox2poly":
        bbox2poly(a.labels, a.out)
    elif a.cmd == "validate":
        raise SystemExit(0 if validate(a.labels, a.nc) else 1)
    elif a.cmd == "stats":
        stats(a.labels)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()

"""
Gera as figuras de resultado da odometria visual para o relatorio.

Le results/trajectories.npz (produzido por plot_from_html.py) e escreve PNGs
em alta resolucao. Tambem imprime as metricas comparativas citadas no texto.

Uso: python gerar_figuras_relatorio.py [pasta_de_saida]
"""
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "results/trajectories.npz"
COLORS = {"ORB": "tab:blue", "SIFT": "tab:green", "KLT": "tab:red"}
DPI = 200


def carregar():
    d = np.load(NPZ)
    return {k: d[k] for k in ["ORB", "SIFT", "KLT"] if k in d}


def _decorar(ax, la, lb, titulo):
    ax.set_title(titulo)
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def _traco(ax, tr, nome, a, b):
    c = COLORS[nome]
    ax.plot(tr[:, a], tr[:, b], color=c, lw=1.4, label=nome, alpha=0.9)
    ax.scatter(tr[0, a], tr[0, b], color=c, marker="o", s=55,
               edgecolor="k", zorder=5)
    ax.scatter(tr[-1, a], tr[-1, b], color=c, marker="X", s=70,
               edgecolor="k", zorder=5)


def fig_mapa(trajs, out):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for nome, tr in trajs.items():
        _traco(ax, tr, nome, 0, 2)
    _decorar(ax, "X (unidades arbitrarias)", "Z (unidades arbitrarias)",
             "Vista de mapa (X-Z) - circulo: inicio, X: fim")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  {out}")


def fig_projecoes(trajs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    proj = [(0, 2, "Topo: X vs Z (vista de mapa)", "X", "Z"),
            (0, 1, "Frontal: X vs Y", "X", "Y"),
            (2, 1, "Lateral: Z vs Y", "Z", "Y")]
    for ax, (a, b, titulo, la, lb) in zip(axes, proj):
        for nome, tr in trajs.items():
            _traco(ax, tr, nome, a, b)
        _decorar(ax, la, lb, titulo)
    fig.suptitle("Trajetorias estimadas por ORB, SIFT e KLT (escala arbitraria)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  {out}")


def fig_3d(trajs, out):
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for nome, tr in trajs.items():
        c = COLORS[nome]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=1.3, label=nome)
        ax.scatter(*tr[0], color=c, marker="o", s=55, edgecolor="k")
        ax.scatter(*tr[-1], color=c, marker="X", s=70, edgecolor="k")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajetoria 3D (escala arbitraria)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  {out}")


def fig_divergencia(trajs, out):
    """Distancia entre pares de metodos ao longo das poses (mede a deriva)."""
    n = min(len(t) for t in trajs.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    estilos = {("ORB", "SIFT"): "tab:purple",
               ("ORB", "KLT"): "tab:orange",
               ("SIFT", "KLT"): "tab:cyan"}
    for a, b in combinations(trajs.keys(), 2):
        d = np.linalg.norm(trajs[a][:n] - trajs[b][:n], axis=1)
        cor = estilos.get((a, b), "gray")
        ax.plot(d, color=cor, lw=1.5, label=f"{a} vs {b} (media {d.mean():.0f})")
    ax.set_xlabel("Indice da pose (keyframe)")
    ax.set_ylabel("Distancia entre trajetorias (unidades arbitrarias)")
    ax.set_title("Divergencia entre metodos ao longo do video (deriva acumulada)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  {out}")


def metricas(trajs):
    print("\nMetricas comparativas:")
    for nome, tr in trajs.items():
        print(f"  {nome}: {len(tr)} poses | fim={np.round(tr[-1], 1)} | "
              f"Y max={tr[:, 1].max():.0f}")
    n = min(len(t) for t in trajs.values())
    for a, b in combinations(trajs.keys(), 2):
        d = np.linalg.norm(trajs[a][:n] - trajs[b][:n], axis=1)
        fim = np.linalg.norm(trajs[a][-1] - trajs[b][-1])
        print(f"  {a} vs {b}: media={d.mean():.0f} | distancia final={fim:.0f}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/figuras_relatorio"
    os.makedirs(out_dir, exist_ok=True)

    trajs = carregar()
    print(f"Gerando figuras em {out_dir}/")
    fig_mapa(trajs, f"{out_dir}/odometria_vista_mapa_XZ.png")
    fig_projecoes(trajs, f"{out_dir}/odometria_projecoes.png")
    fig_3d(trajs, f"{out_dir}/odometria_3d.png")
    fig_divergencia(trajs, f"{out_dir}/odometria_divergencia.png")
    metricas(trajs)

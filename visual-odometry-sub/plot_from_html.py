"""
Extrai as trajetorias 3D de um HTML gerado pelo Plotly (main.py) e gera:
  - um PNG estatico com 3 projecoes 2D + resumo
  - um .npz reutilizavel (para replotar sem reprocessar o video)

Uso: python plot_from_html.py results/resultado_vo_6.html
"""
import sys
import re
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decode_bdata(b64):
    b64 = b64.replace("\\u002f", "/").replace("\\u002b", "+")
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype="<f8")


def extract_trajectories(html_path):
    html = open(html_path, encoding="utf-8").read()
    pat = re.compile(
        r'"x":\{"dtype":"f8","bdata":"([^"]*)"\},'
        r'"y":\{"dtype":"f8","bdata":"([^"]*)"\},'
        r'"z":\{"dtype":"f8","bdata":"([^"]*)"\},'
        r'"type":"scatter3d"'
    )
    names = ["ORB", "SIFT", "KLT"]
    trajs = {}
    for i, m in enumerate(pat.finditer(html)):
        x = decode_bdata(m.group(1))
        y = decode_bdata(m.group(2))
        z = decode_bdata(m.group(3))
        name = names[i] if i < len(names) else f"traj_{i}"
        trajs[name] = np.column_stack([x, y, z])
    return trajs


def plot(trajs, out_png):
    colors = {"ORB": "tab:blue", "SIFT": "tab:green", "KLT": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    projections = [
        (0, 2, "Topo: X vs Z (vista de mapa)", "X", "Z"),
        (0, 1, "Frontal: X vs Y", "X", "Y"),
        (2, 1, "Lateral: Z vs Y", "Z", "Y"),
    ]
    for ax, (a, b, title, la, lb) in zip(axes, projections):
        for name, tr in trajs.items():
            c = colors.get(name, "gray")
            ax.plot(tr[:, a], tr[:, b], color=c, lw=1.3, label=name, alpha=0.85)
            ax.scatter(tr[0, a], tr[0, b], color=c, marker="o", s=45,
                       edgecolor="k", zorder=5)
            ax.scatter(tr[-1, a], tr[-1, b], color=c, marker="X", s=55,
                       edgecolor="k", zorder=5)
        ax.set_title(title)
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Trajetorias VO (escala arbitraria) - circulo=inicio, X=fim",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"PNG salvo em: {out_png}")


if __name__ == "__main__":
    html_path = sys.argv[1] if len(sys.argv) > 1 else "results/resultado_vo_6.html"
    trajs = extract_trajectories(html_path)
    for name, tr in trajs.items():
        print(f"{name}: {len(tr)} pontos | fim={np.round(tr[-1], 2)}")

    np.savez("results/trajectories.npz", **trajs)
    print("Dados salvos em: results/trajectories.npz")

    plot(trajs, "results/trajetorias.png")

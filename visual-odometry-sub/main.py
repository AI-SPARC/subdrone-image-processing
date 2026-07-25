import os

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vo import VisualOdometry
from utils import preprocess_frame

# ===================== CONFIGURACOES =====================
VIDEO_PATH = "C:/Users/guilh/Projetos/subdrone-image-processing/visual-odometry-sub/dataset/processed/Black Eder_090419_costelas do navio_2_vo.mp4"

# Distancia (em frames) entre keyframes. A 30fps, FRAME_STEP=3 ~ 0.1s.
# Baseline maior = movimento maior entre frames = matriz essencial mais
# estavel. Se ficar ruidoso, aumente; se pular demais, diminua.
FRAME_STEP = 3

# Apos MAX_SKIP falhas seguidas, ressincroniza o keyframe (avanca prev_gray
# para o frame atual). Evita que um trecho ruim (virada/turbidez) congele o
# keyframe no passado e trave a odometria pelo resto do video.
MAX_SKIP = 2

# Limite de frames a processar (None = video inteiro).
MAX_FRAMES = None

# fx = fy = FX_SCALE * largura. Isto e um CHUTE: o ideal e calibrar a
# camera (de preferencia dentro d'agua) e substituir K por valores reais.
FX_SCALE = 1.0

METHODS = ["ORB", "SIFT", "KLT"]
COLORS = {"ORB": "blue", "SIFT": "green", "KLT": "red"}
# ========================================================


def get_output_path():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    counter = 1
    while os.path.exists(os.path.join(results_dir, f"resultado_vo_{counter}.html")):
        counter += 1
    return os.path.join(results_dir, f"resultado_vo_{counter}.html")


def run_method(method):
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if not ret:
        raise Exception("Erro ao ler video.")

    prev_gray = preprocess_frame(frame)

    h, w = prev_gray.shape[:2]
    fx = fy = FX_SCALE * w
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    vo = VisualOdometry(K, method=method)

    # A trajetoria SEMPRE comeca na origem (todos os metodos partem de 0,0,0).
    trajectory = [(0.0, 0.0, 0.0, 0)]

    frame_idx = 0
    n_ok = 0
    n_fail = 0
    consec_fail = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        if frame_idx % FRAME_STEP != 0:
            continue

        gray = preprocess_frame(frame)
        t, _, ok = vo.process_frame(prev_gray, gray)

        if ok:
            # So registra e avanca o keyframe quando a estimativa foi confiavel.
            trajectory.append((t[0][0], t[1][0], t[2][0], frame_idx))
            prev_gray = gray
            consec_fail = 0
            n_ok += 1
        else:
            # Falha transiente: mantem prev_gray por ate MAX_SKIP tentativas
            # (baseline cresce e tende a estabilizar). Se persistir, ressincroniza
            # para nao travar a odometria no resto do video.
            n_fail += 1
            consec_fail += 1
            if consec_fail >= MAX_SKIP:
                prev_gray = gray
                consec_fail = 0

        if frame_idx % 90 == 0:
            print(f"  [{method}] frame {frame_idx} | ok={n_ok} fail={n_fail}")

        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break

    cap.release()
    print(f"  [{method}] concluido: {n_ok} keyframes validos, {n_fail} descartados")
    return np.array(trajectory)


def main():
    trajectories = {}
    for method in METHODS:
        print(f"\nRodando metodo: {method}")
        trajectories[method] = run_method(method)

    # ------- resumo textual -------
    for method in METHODS:
        traj = trajectories[method]
        mid = len(traj) // 2
        print(f"\n{method}: {len(traj)} pontos")
        print(f"  inicio: {np.round(traj[0][:3], 3)}")
        print(f"  meio:   {np.round(traj[mid][:3], 3)}")
        print(f"  fim:    {np.round(traj[-1][:3], 3)}")

    # ------- plot: 3D + 3 projecoes 2D -------
    # As projecoes ajudam a identificar qual plano corresponde ao "mapa"
    # real (depende da orientacao da camera: para frente vs para baixo).
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("3D (X, Y, Z)", "Topo: X vs Z",
                        "Frontal: X vs Y", "Lateral: Z vs Y"),
    )

    for method in METHODS:
        traj = trajectories[method]
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        c = COLORS[method]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=method,
            line=dict(width=4, color=c), legendgroup=method,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=z, mode="lines", name=method, line=dict(color=c),
            legendgroup=method, showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=method, line=dict(color=c),
            legendgroup=method, showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=z, y=y, mode="lines", name=method, line=dict(color=c),
            legendgroup=method, showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        title="Odometria visual - ORB vs SIFT vs KLT (escala arbitraria)",
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    for r, col in [(1, 2), (2, 1), (2, 2)]:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=r, col=col)

    output_path = get_output_path()
    fig.write_html(output_path)
    print(f"\nGrafico salvo em: {output_path}")


if __name__ == "__main__":
    main()

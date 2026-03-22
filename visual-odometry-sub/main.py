import cv2
import numpy as np
import plotly.graph_objects as go
from vo import VisualOdometry
from utils import preprocess_frame

video_path = "D:/guilh/Downloads/pibiti mn2001/visual-odometry-sub/dataset/processed/Black Eder_090419_costelas do navio_2_vo.mp4"
SCALE = 0.5  # Mude para 1.0 para resolução original

methods = ["ORB", "SIFT", "KLT"]
trajectories = {}

for method in methods:
    print(f"\nRodando método: {method}")

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        raise Exception("Erro ao ler vídeo.")

    prev_gray = preprocess_frame(frame)

    # Matriz intrínseca baseada no frame preprocessado
    fx = 800 * SCALE
    fy = 800 * SCALE
    cx = (prev_gray.shape[1] / 2)
    cy = (prev_gray.shape[0] / 2)


    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    vo = VisualOdometry(K, method=method)

    trajectory = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w*SCALE), int(h*SCALE)))

        gray = preprocess_frame(frame)

        t = vo.process_frame(prev_gray, gray)

        trajectory.append((t[0][0], t[1][0], t[2][0]))

        prev_gray = gray
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"{method} processando frame {frame_count}")

        if frame_count > 300:
            break


    cap.release()

    trajectories[method] = np.array(trajectory)

# ================== PLOT 3D ==================

fig = go.Figure()

colors = {
    "ORB": "blue",
    "SIFT": "green",
    "KLT": "red"
}

for method in methods:
    traj = trajectories[method]

    fig.add_trace(go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        name=method,
        line=dict(width=4, color=colors[method])
    ))

fig.update_layout(
    title="Comparação 3D - ORB vs SIFT vs KLT",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode='data'
    )
)

fig.show()





"""
import cv2
import numpy as np
import plotly.graph_objects as go
from vo import VisualOdometry
from utils import preprocess_frame

#video_path = r"D:\guilh\Downloads\pibiti mn2001\visual-odometry-sub\dataset\Black Eder_090419_costelas do navio_2.MP4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
prev_gray = preprocess_frame(frame)

# ===== MATRIZ INTRÍNSECA (exemplo) =====
fx = fy = 800
cx = prev_gray.shape[1] / 2
cy = prev_gray.shape[0] / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

vo = VisualOdometry(K, method="KLT")

trajectory = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray = preprocess_frame(frame)

    t = vo.process_frame(prev_gray, gray)

    trajectory.append((t[0][0], t[1][0], t[2][0]))

    prev_gray = gray

cap.release()

trajectory = np.array(trajectory)

# ===== Plot 3D interativo =====
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=trajectory[:, 0],
    y=trajectory[:, 1],
    z=trajectory[:, 2],
    mode='lines',
    line=dict(width=4)
))

fig.update_layout(
    title="3D Visual Odometry Trajectory",
    scene=dict(aspectmode='data')
)

fig.show()
"""



"""""
# ================== PLOT 3D ==================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visual Odometry Trajectory")

# Ajuste automático de escala para não distorcer
max_range = np.array([
    trajectory[:,0].max()-trajectory[:,0].min(),
    trajectory[:,1].max()-trajectory[:,1].min(),
    trajectory[:,2].max()-trajectory[:,2].min()
]).max() / 2.0

mid_x = (trajectory[:,0].max()+trajectory[:,0].min()) * 0.5
mid_y = (trajectory[:,1].max()+trajectory[:,1].min()) * 0.5
mid_z = (trajectory[:,2].max()+trajectory[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

"""


""""
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Matriz intrínseca (exemplo)
fx = fy = 800
cx = frame.shape[1] / 2
cy = frame.shape[0] / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

vo = VisualOdometry(K, method="KLT")  # ORB, SIFT ou KLT

trajectory = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t = vo.process_frame(prev_gray, gray)
    trajectory.append((t[0][0], t[2][0]))

    prev_gray = gray

cap.release()

trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Trajectory")
plt.show()
"""
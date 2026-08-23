from ultralytics import YOLO
import torch
import os
import random
import numpy as np
from datetime import datetime
import shutil

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_YAML = "data.yaml"
PHASE1_BEST = (
    "runs/detect/lionfish_detection/"
    "train_20260207_2139_phase1/weights/best.pt"
)
PROJECT_NAME = "lionfish_detection"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}_phase2"
IMG_SIZE = 640
EPOCHS = 50
BATCH = 2
DEVICE = "cpu"
FINAL_MODEL_DIR = "trained_models/yolov12_lionfish"
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

model = YOLO(PHASE1_BEST)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT_NAME,
    name=RUN_NAME,
    freeze=0,
    optimizer="AdamW",
    lr0=1e-4,
    lrf=0.01,
    weight_decay=5e-4,
    warmup_epochs=3,
    warmup_bias_lr=0.05,
    warmup_momentum=0.8,
    cos_lr=True,
    amp=False,
    patience=40,
    save=True,
    save_period=10,
    hsv_h=0.02,
    hsv_s=0.6,
    hsv_v=0.5,
    mosaic=0.5,
    mixup=0.05,
    close_mosaic=20,
    workers=0,
    pretrained=True,
    val=True,
    plots=True,
    save_json=True,
    verbose=True,
)

BEST_MODEL_PATH = os.path.join(results.save_dir, "weights", "best.pt")
LAST_MODEL_PATH = os.path.join(results.save_dir, "weights", "last.pt")

shutil.copy(BEST_MODEL_PATH, FINAL_MODEL_DIR + "/best.pt")
shutil.copy(LAST_MODEL_PATH, FINAL_MODEL_DIR + "/last.pt")

print("Best:", BEST_MODEL_PATH)
print("Last:", LAST_MODEL_PATH)

model = YOLO(BEST_MODEL_PATH)

print("\nExportando...\n")

model.export(format="onnx")
model.export(format="torchscript")


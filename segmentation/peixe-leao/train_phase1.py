from ultralytics import YOLO
import torch
import os
import random
import numpy as np
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_YAML = "data.yaml"
MODEL_NAME = "yolo12n.pt"
PROJECT_NAME = "lionfish_detection"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}_phase1"
IMG_SIZE = 640
EPOCHS = 30
BATCH = 2
DEVICE = "cpu"

model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT_NAME,
    name=RUN_NAME,
    freeze=10,
    optimizer="AdamW",
    lr0=3e-4,
    lrf=0.01,
    weight_decay=5e-4,
    warmup_epochs=5,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
    cos_lr=True,
    amp=False,
    patience=30,
    save=True,
    save_period=10,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.6,
    degrees=5,
    translate=0.05,
    scale=0.5,
    shear=2,
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.15,
    mosaic=1.0,
    mixup=0.2,
    close_mosaic=30,
    workers=2,
    pretrained=True,
    val=True,
    plots=True,
    save_json=True,
    verbose=True,
)

BEST_PATH = os.path.join(results.save_dir, "weights", "best.pt")

print("Best model:", BEST_PATH)


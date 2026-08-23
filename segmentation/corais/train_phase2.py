"""Fase 2 do treino de segmentacao de corais, com ajuste fino completo.

Parte do melhor peso da fase 1, descongela a rede e usa taxa de aprendizado
menor. No final copia os pesos para trained_models/ e exporta em ONNX e
TorchScript.
"""

import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

from train_common import (PROJECT_NAME, encontrar_peso_fase1,
                          escolher_dispositivo, resumo)

SEED = 42
DATA_YAML = "data.yaml"
IMG_SIZE = 640
FINAL_MODEL_DIR = "trained_models/yolo_coral_seg"


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ap = argparse.ArgumentParser(
        description="Fase 2 do treino (fine-tuning total)")
    ap.add_argument("--peso", default=None, help="best.pt da fase 1")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    phase1_best = encontrar_peso_fase1(args.peso)

    device, batch, amp = escolher_dispositivo(
        preferir_xpu=bool(args.device and args.device.startswith("xpu")))
    if args.device:
        device = args.device
    if args.batch:
        batch = args.batch
    epochs = args.epochs if args.epochs else (3 if args.rapido else 50)
    workers = args.workers if args.workers is not None else 4

    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}_phase2"
    resumo("Fase 2 (fine-tuning completo)", device, batch, amp, epochs, IMG_SIZE)

    model = YOLO(phase1_best)

    results = model.train(
        data=DATA_YAML,
        task="segment",
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        device=device,
        project=PROJECT_NAME,
        name=run_name,
        freeze=0,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=1 if args.rapido else 3,
        warmup_bias_lr=0.05,
        warmup_momentum=0.8,
        cos_lr=True,
        amp=amp,
        patience=40,
        save=True,
        save_period=10,
        fraction=0.05 if args.rapido else 1.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.5,
        mixup=0.05,
        close_mosaic=20,
        workers=workers,
        pretrained=True,
        val=True,
        plots=True,
        # Em segmentacao o save_json acumula mascaras em resolucao original e
        # estoura a memoria na validacao.
        save_json=False,
        max_det=100,
        verbose=True,
    )

    best_path = os.path.join(results.save_dir, "weights", "best.pt")
    last_path = os.path.join(results.save_dir, "weights", "last.pt")

    shutil.copy(best_path, FINAL_MODEL_DIR + "/best.pt")
    shutil.copy(last_path, FINAL_MODEL_DIR + "/last.pt")

    print("Best:", best_path)
    print("Last:", last_path)

    model = YOLO(best_path)
    print("\nExportando...\n")
    model.export(format="onnx")
    model.export(format="torchscript")

    print("\nProximo passo - avaliar (olhe o recall de coral_branqueado):")
    print(f"  python evaluate.py val --model {FINAL_MODEL_DIR}/best.pt "
          f"--data data.yaml --split test")


if __name__ == "__main__":
    main()

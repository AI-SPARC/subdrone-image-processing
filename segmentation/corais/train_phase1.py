"""Fase 1 do treino de segmentacao de corais, com o backbone congelado.

O treino fica dentro de main() sob if __name__ == "__main__" porque no Windows
o DataLoader usa spawn e reimporta este arquivo em cada worker.
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

from train_common import PROJECT_NAME, escolher_dispositivo, resumo

SEED = 42
DATA_YAML = "data.yaml"
MODEL_NAME = "yolo11n-seg.pt"
IMG_SIZE = 640


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ap = argparse.ArgumentParser(
        description="Fase 1 do treino (backbone congelado)")
    ap.add_argument("--rapido", action="store_true",
                    help="3 epocas em 5%% dos dados, so para validar o fluxo")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="ex.: xpu, 0, cpu")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    device, batch, amp = escolher_dispositivo(
        preferir_xpu=bool(args.device and args.device.startswith("xpu")))
    if args.device:
        device = args.device
    if args.batch:
        batch = args.batch
    epochs = args.epochs if args.epochs else (3 if args.rapido else 30)
    workers = args.workers if args.workers is not None else 4

    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}_phase1"
    resumo("Fase 1 (backbone congelado)", device, batch, amp, epochs, IMG_SIZE)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        task="segment",
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        device=device,
        project=PROJECT_NAME,
        name=run_name,
        freeze=10,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=1 if args.rapido else 5,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        cos_lr=True,
        amp=amp,
        patience=30,
        save=True,
        save_period=10,
        fraction=0.05 if args.rapido else 1.0,
        # Aumentos de cor moderados: exagerar prejudicaria a leitura de saude.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.5,
        shear=2,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.15,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=30,
        workers=workers,
        pretrained=True,
        val=True,
        plots=True,
        # Em segmentacao o save_json acumula mascaras em resolucao original e
        # estoura a memoria na validacao. Use evaluate.py val para gera-lo.
        save_json=False,
        max_det=100,
        verbose=True,
    )

    best = os.path.join(results.save_dir, "weights", "best.pt")
    print("\nBest model:", best)
    print("\nProximo passo (encontra este peso sozinho):")
    print("  python train_phase2.py")


if __name__ == "__main__":
    main()

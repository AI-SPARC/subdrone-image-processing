"""Configuracao compartilhada pelos dois scripts de treino."""

from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_NAME = "coral_segmentation"


def _desativar_foreach_no_xpu() -> None:
    """Forca os otimizadores ao caminho tensor-a-tensor, opcional em GPU Intel.

    Evita o erro OUT_OF_RESOURCES em torch._foreach_sqrt, ao custo de deixar o
    treino cerca de 19x mais lento.
    """
    for nome in ("AdamW", "Adam", "SGD", "RMSprop"):
        classe = getattr(torch.optim, nome, None)
        if classe is None or getattr(classe, "_foreach_desligado", False):
            continue

        class SemForeach(classe):  # type: ignore[misc, valid-type]
            _foreach_desligado = True

            def __init__(self, *a, **kw):
                kw.setdefault("foreach", False)
                super().__init__(*a, **kw)

        SemForeach.__name__ = nome
        SemForeach.__qualname__ = nome
        setattr(torch.optim, nome, SemForeach)


def escolher_dispositivo(preferir_xpu: bool = False) -> tuple[str, int, bool]:
    """Devolve (device, batch, amp) conforme o hardware disponivel.

    A GPU Intel integrada nao e escolhida por padrao: ela conclui treinos em
    subconjuntos pequenos mas falha no dataset completo, com erros do backend
    Level Zero. Use --device xpu ou CORAL_USE_XPU=1 para tentar.
    """
    quer_xpu = preferir_xpu or os.environ.get("CORAL_USE_XPU") == "1"
    tem_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if tem_xpu and quer_xpu:
        print(f"Dispositivo: XPU (Intel) - {torch.xpu.get_device_name(0)}")
        if os.environ.get("CORAL_XPU_NO_FOREACH") == "1":
            _desativar_foreach_no_xpu()
        amp = os.environ.get("CORAL_XPU_AMP") == "1"
        return "xpu", 4, amp

    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Dispositivo: CUDA - {nome} ({vram:.0f} GB)")
        return "0", (16 if vram >= 10 else 8), True

    print("Dispositivo: CPU")
    return "cpu", 8, False


def encontrar_peso_fase1(explicito: str | None = None) -> str:
    """Localiza o best.pt da fase 1 mais recente."""
    if explicito:
        if not os.path.isfile(explicito):
            raise SystemExit(f"peso da fase 1 nao encontrado: {explicito}")
        return explicito

    candidatos: list[Path] = []
    for raiz in (Path(PROJECT_NAME), Path("runs") / "segment" / PROJECT_NAME,
                 Path("runs") / "segment"):
        if raiz.is_dir():
            candidatos.extend(raiz.glob("*phase1*/weights/best.pt"))

    if not candidatos:
        raise SystemExit(
            "Nao achei o best.pt da fase 1.\n"
            "Rode 'python train_phase1.py' antes, ou aponte o caminho:\n"
            "  python train_phase2.py --peso caminho/para/best.pt"
        )

    melhor = max(candidatos, key=lambda p: p.stat().st_mtime)
    print(f"Peso da fase 1 (mais recente): {melhor}")
    return str(melhor)


def resumo(fase: str, device: str, batch: int, amp: bool, epochs: int,
           imgsz: int) -> None:
    print()
    print(f"=== Segmentacao de corais - {fase} ===")
    print(f"  device={device}  batch={batch}  amp={amp}  "
          f"epochs={epochs}  imgsz={imgsz}")
    print("  3 classes: coral_vivo, coral_branqueado, coral_morto")
    print()

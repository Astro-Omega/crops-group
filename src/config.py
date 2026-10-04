import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PredictionConfig:
    # Arquitectura
    hidden_layers: tuple = (128, 64, 32)
    activation: str = "relu"
    dropout: float = 0.2
    l2: float = 1e-4
    # Entrenamiento
    learning_rate: float = 1e-3
    epochs: int = 300
    batch_size: int = 32
    patience: int = 25            # early stopping
    # Validación: "temporal" usa el último año del histórico como validación
    # (lo más parecido a predecir 2025); "random" usa una fracción aleatoria.
    val_strategy: str = "temporal"
    val_fraction: float = 0.15    # solo aplica a "random"
    refit_full: bool = True       # reentrena con todo el histórico usando la mejor época
    # Objetivo
    log_target: bool = True       # log1p(rendimiento): estabiliza distribuciones sesgadas
    seed: int = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "output"))
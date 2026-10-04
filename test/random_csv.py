# ¡Importante! aquí se generan CSV con datos ficticios, por ende solo sirven para realizar pruebas
# tener mucho cuidado al usarlo como material de entrenamiento o de predicción  

# Se basa en la generación de CSVs de prueba con el nombre 'sample_test_2025.csv'

# Prontamente se designará al README para dejar de acomular basura
# Uso (desde la raíz del repo):
#    python test/random_csv.py                  <-- sample_train_2021_2024.csv + sample_test_2025.csv
#    python test/random_csv.py --no-target      <-- test sin Rendimiento (solo predecir)
#    python test/random_csv.py --test-year 2026 --seed 7#

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Diccionario subregion (key) | array de municipios y ciudades (value)
SUB = {
    "Sabanas": ["Buenavista", "Corozal", "El Roble", "Galeras", "Sampués", "San Juan de Betulia",
                "San Pedro", "San Luis de Sincé", "Sincelejo"],
    "Golfo de Morrosquillo": ["Coveñas", "San Antonio de Palmito", "San Onofre", "Santiago de Tolú", "Toluviejo"],
    "Montes de María": ["Chalán", "Colosó", "Los Palmitos", "Morroa", "Ovejas"],
    "Mojana": ["Guaranda", "Majagual", "Sucre"],
    "San Jorge": ["Caimito", "La Unión", "San Benito Abad", "San Marcos"],
}

# cultivo: (rendimiento base t/ha, periodos posibles)
CROPS = {"Maíz Tradicional": (2.0, ["Semestre A", "Semestre B"]), "Maíz Mecanizado": (4.0, ["Semestre A", "Semestre B"]),
            "Arroz Secano Mecanizado": (3.7, ["Semestre A", "Semestre B"]), "Ajonjolí": (0.8, ["Semestre B"]),
            "Yuca Dulce": (8.0, ["Anual"]), "Ñame": (9.0, ["Anual"]), "Ahuyama": (5.0, ["Anual"]),
            "Plátano": (6.0, ["Permanente"]), "Cacao": (0.6, ["Permanente"]), "Mango": (8.0, ["Permanente"])}

# subregiones de sucre
SUBF = {"Sabanas": 1.0, "Golfo de Morrosquillo": 0.95, "Montes de María": 1.1, "Mojana": 0.9, "San Jorge": 1.05}

# Mismo esquema que Estados_Cultivos_Sucre_Limpio_2021_2024.csv
COLS = ["Año", "Cultivo", "Periodo", "Municipios", "Subregión", "Área Sembrada (Has)",
        "Área Cosechada (Has)", "Rendimiento promedio (Ton/Ha)", "Producción (Ton)"]
HIDDEN_IN_TEST = ["Área Cosechada (Has)", "Rendimiento promedio (Ton/Ha)", "Producción (Ton)"]

def build(years, with_target: bool = True, seed: int = 0) -> pd.DataFrame:
    #with_target=False imita el CSV de 2025 antes de la cosecha: solo datos de siembra
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for sub, munis in SUB.items():
            for m in munis:
                for crop in rng.choice(list(CROPS), size=rng.integers(4, 8), replace=False):
                    base, periods = CROPS[crop]
                    for per in periods:
                        area = float(np.round(rng.lognormal(4, 1), 1))
                        harv = float(np.round(area * rng.choice([1.0, 1.0, 0.9, 0.5]), 1))
                        r = base * SUBF[sub] * (1 + 0.02 * (y - 2021)) * (1 + 0.05 * np.log(area) - 0.2)
                        r = round(max(r * rng.normal(1, 0.08), 0.05), 2)
                        rows.append([y, crop, per, m, sub, area, harv, r, round(r * harv, 2)])
    df = pd.DataFrame(rows, columns=COLS)
    return df if with_target else df.drop(columns=HIDDEN_IN_TEST)

# Simula el CSV del año siguiente usando la estructura REAL (mismos cultivos, periodos,
# municipios y subregiones del último año del histórico). Solo varía el área y el rendimiento
# con ruido pequeño. Sirve para probar el flujo con categorías que sí existen en tu dataset.

def from_real(real: pd.DataFrame, year: int = 2025, with_target: bool = True, seed: int = 0) -> pd.DataFrame:
    if list(real.columns) != COLS:
        raise ValueError(f"El CSV real debe tener las columnas {COLS}")
    rng = np.random.default_rng(seed)
    base = real[real["Año"] == real["Año"].max()].copy()
    n = len(base)
    base["Año"] = year
    base["Área Sembrada (Has)"] = (base["Área Sembrada (Has)"] * rng.lognormal(0, 0.15, n)).round(1).clip(lower=0.25)
    ratio = (base["Área Cosechada (Has)"] / base["Área Sembrada (Has)"].replace(0, np.nan)).fillna(1.0)
    base["Área Cosechada (Has)"] = (base["Área Sembrada (Has)"] * ratio.clip(upper=1.0)).round(1)
    base["Rendimiento promedio (Ton/Ha)"] = (base["Rendimiento promedio (Ton/Ha)"] * rng.normal(1, 0.05, n)).round(2).clip(lower=0.05)
    base["Producción (Ton)"] = (base["Rendimiento promedio (Ton/Ha)"] * base["Área Cosechada (Has)"]).round(2)
    base = base.reset_index(drop=True)
    return base if with_target else base.drop(columns=HIDDEN_IN_TEST)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", nargs=2, type=int, default=[2021, 2024], metavar=("DESDE", "HASTA"))
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--no-target", action="store_true", help="el CSV de test solo trae datos de siembra (sin Rendimiento, Cosechada ni Producción)")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-real", type=Path, default=None, metavar="CSV",
                    help="genera SOLO el test a partir de tu CSV real (mismas categorías); no crea train")
    a = ap.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.from_real:
        real = pd.read_csv(a.from_real, encoding="utf-8-sig")
        f_test = a.outdir / f"sample_test_{a.test_year}.csv"
        from_real(real, a.test_year, with_target=not a.no_target, seed=a.seed).to_csv(f_test, index=False)
        print(f"OK (a partir de {a.from_real.name}) -> {f_test}")
        return
    y0, y1 = a.train_years
    f_train = a.outdir / f"sample_train_{y0}_{y1}.csv"
    f_test = a.outdir / f"sample_test_{a.test_year}.csv"
    build(range(y0, y1 + 1), seed=a.seed).to_csv(f_train, index=False)
    build([a.test_year], with_target=not a.no_target, seed=a.seed + 1).to_csv(f_test, index=False)
    print(f"OK -> {f_train}\n   -> {f_test}")

if __name__ == "__main__":
    main()
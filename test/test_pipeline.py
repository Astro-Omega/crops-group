# Apartado orientado tipo test para el pipeline 
import numpy as np
import pytest  # PROBLEMA AVISMAL, que wea descargar otra dependencia, haganlo ustedes 

from src.config import PredictionConfig
from src.pipeline import read_csv, run_pipeline, save_predictions
from random_csv import build

FAST = PredictionConfig(hidden_layers=(16, 8), epochs=8, patience=3, seed=1)

@pytest.fixture(scope="module")
def train():
    return build(range(2021, 2025), seed=0)

def test_prediction_contract(train):
    test = build([2025], seed=1)
    res = run_pipeline(train, test, FAST)
    out = res["predictions"]
    assert len(out) == len(test)
    for col in ["Rendimiento_Predicho", "Produccion_Estimada_t", "Rendimiento_Real", "Error_Abs"]:
        assert col in out.columns
    assert (out["Rendimiento_Predicho"] >= 0).all()
    # con Área Cosechada en el test, la producción usa esa área
    assert np.allclose(out["Produccion_Estimada_t"], out["Rendimiento_Predicho"] * out["Area_Cosechada_Usada"], rtol=1e-3)
    assert res["test_metrics"] is not None

def test_test_without_target(train):
    out = run_pipeline(train, build([2025], with_target=False, seed=1), FAST)["predictions"]
    assert "Rendimiento_Real" not in out.columns
    assert out["Rendimiento_Predicho"].notna().all()
    # sin Área Cosechada en el test, se estima con la razón histórica (nunca NaN)
    assert out["Area_Cosechada_Usada"].notna().all() and (out["Produccion_Estimada_t"] >= 0).all()

def test_unseen_municipio_does_not_break(train):
    test = build([2025], seed=1)
    test.loc[0, "Municipio"] = "Municipio Nuevo"
    out = run_pipeline(train, test, FAST)["predictions"]
    assert out["Rendimiento_Predicho"].notna().all()

def test_csv_roundtrip(train, tmp_path):
    train.to_csv(tmp_path / "train.csv", index=False)
    build([2025], seed=1).to_csv(tmp_path / "test.csv", index=False)
    res = run_pipeline(read_csv(tmp_path / "train.csv"), read_csv(tmp_path / "test.csv"), FAST)  # ruta -> DataFrame
    path = save_predictions(res["predictions"], tmp_path / "out" / "pred.csv")
    assert path.exists() and len(read_csv(path)) == len(res["predictions"])

def test_real_schema_columns_are_recognised(train):
    # Año separado de Periodo (categórico) y 'Municipios' en plural no deben chocar
    from src.pipeline import clean
    df = clean(train)
    assert {"Anio", "Periodo", "Municipio", "Subregion", "Area_Sembrada", "Rendimiento"} <= set(df.columns)
    assert set(df["Periodo"].unique()) <= {"ANUAL", "SEMESTRE A", "SEMESTRE B", "PERMANENTE"}
    assert df["Anio"].between(2021, 2024).all()

def test_from_real_keeps_categories_and_schema(train):
    from random_csv import COLS, HIDDEN_IN_TEST, from_real
    real = build(range(2021, 2025), seed=0)
    full, siembra = from_real(real, 2025), from_real(real, 2025, with_target=False)
    assert list(full.columns) == COLS and set(HIDDEN_IN_TEST).isdisjoint(siembra.columns)
    assert (full["Año"] == 2025).all() and len(full) == (real["Año"] == 2024).sum()
    for col in ["Cultivo", "Periodo", "Municipios", "Subregión"]:
        assert set(full[col]) <= set(real[col])

from __future__ import annotations

import argparse
import io
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import OUTPUT_DIR, PredictionConfig

BASE_CAT = ["Cultivo", "Periodo", "Municipio", "Subregion"]   # Periodo = Anual / Semestre A / Semestre B / Permanente
OPT_CAT = ["Ciclo"]  # solo se usa si el CSV la trae
REQUIRED = ["Cultivo", "Periodo", "Municipio", "Subregion", "Area_Sembrada", "Anio"]
NUM_COLS = ["Area_log", "Anio"]

# Nombres normalizados (sin tildes, minúsculas, símbolos -> "_") -> nombre interno
_ALIASES = {
    "cultivo": "Cultivo", "periodo": "Periodo",
    "ano": "Anio", "anio": "Anio", "year": "Anio",
    "municipio": "Municipio", "municipios": "Municipio",
    "subregion": "Subregion", "sub_region": "Subregion",
    "area_sembrada": "Area_Sembrada", "area_sembrada_ha": "Area_Sembrada", "area_sembrada_has": "Area_Sembrada",
    "area_cosechada": "Area_Cosechada", "area_cosechada_ha": "Area_Cosechada", "area_cosechada_has": "Area_Cosechada",
    "rendimiento": "Rendimiento", "rendimiento_t_ha": "Rendimiento", "rendimiento_ton_ha": "Rendimiento",
    "rendimiento_promedio_ton_ha": "Rendimiento",
    "produccion": "Produccion", "produccion_ton": "Produccion", "produccion_t": "Produccion",
    "ciclo": "Ciclo",
}

# ----------------------------------------------------------------------------- carga
def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# Lee un CSV desde ruta o archivo subido; detecta separador y codificación
def read_csv(source) -> pd.DataFrame:
    if hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        with open(source, "rb") as f:
            raw = f.read()
    try:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _ALIASES.get(_norm(c), c) for c in df.columns})
    return df.loc[:, ~df.columns.duplicated()]

def _to_number(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "if":
        return s
    t = s.astype(str).str.strip()
    both = t.str.contains(r"\.", regex=True) & t.str.contains(",", regex=False)
    t = t.where(~both, t.str.replace(".", "", regex=False))  # 1.234,5 -> 1234,5
    return pd.to_numeric(t.str.replace(",", ".", regex=False), errors="coerce")

def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    df = standardize_columns(df).copy()
    if "Anio" not in df.columns and "Periodo" in df.columns:  # respaldo: año dentro de Periodo ("2021A")
        df["Anio"] = pd.to_numeric(df["Periodo"].astype(str).str.extract(r"(20\d{2})")[0], errors="coerce")
    missing = [c for c in REQUIRED if c not in df.columns]
    if require_target and "Rendimiento" not in df.columns:
        missing.append("Rendimiento")
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}. Columnas encontradas: {list(df.columns)}")

    for c in ("Anio", "Area_Sembrada", "Area_Cosechada", "Produccion", "Rendimiento"):
        if c in df.columns:
            df[c] = _to_number(df[c])

    cats = [c for c in BASE_CAT + OPT_CAT if c in df.columns]
    for c in cats:
        df[c] = df[c].astype("string").str.strip().str.upper()

    need = cats + ["Area_Sembrada", "Anio"] + (["Rendimiento"] if require_target else [])
    df = df.dropna(subset=need)
    df = df[df["Area_Sembrada"] >= 0]
    if require_target:
        df = df[df["Rendimiento"] >= 0]
    return df.reset_index(drop=True)

def make_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = df[cat_cols].astype(str).copy()
    X["Area_log"] = np.log1p(df["Area_Sembrada"].astype(float))
    X["Anio"] = df["Anio"].astype(float)
    return X

def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

# ----------------------------------------------------------------------------- modelo
class YieldPredictor:
    # Preprocesamiento + MLP para predecir Rendimiento (t/ha)
    def __init__(self, cfg: PredictionConfig | None = None):
        self.cfg = cfg or PredictionConfig()
        self.model = None
        self.history: dict = {}
        self.best_epoch = 0
        self.val_metrics: dict = {}
        self.train_df: pd.DataFrame | None = None
        self.y_scaler = StandardScaler()

    # -- helpers
    def _build(self, n_in: int):
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers

        c = self.cfg
        m = tf.keras.Sequential([layers.Input(shape=(n_in,))])
        for units in c.hidden_layers:
            m.add(layers.Dense(units, activation=c.activation,
                                kernel_regularizer=regularizers.l2(c.l2)))
            if c.dropout > 0:
                m.add(layers.Dropout(c.dropout))
        m.add(layers.Dense(1))
        m.compile(optimizer=tf.keras.optimizers.Adam(c.learning_rate), loss="mse", metrics=["mae"])
        return m

    def _fwd_y(self, y):
        y = np.asarray(y, float)
        return np.log1p(y) if self.cfg.log_target else y

    def _inv_y(self, y_scaled):
        y = self.y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
        y = np.expm1(y) if self.cfg.log_target else y
        return np.clip(y, 0, None)

    def _val_mask(self, df: pd.DataFrame) -> np.ndarray:
        if self.cfg.val_strategy == "temporal" and df["Anio"].nunique() > 1:
            return (df["Anio"] == df["Anio"].max()).to_numpy()
        rng = np.random.default_rng(self.cfg.seed)
        mask = np.zeros(len(df), bool)
        mask[rng.choice(len(df), max(1, int(len(df) * self.cfg.val_fraction)), replace=False)] = True
        return mask

    # -- entrenamiento
    def fit(self, df_raw: pd.DataFrame, callbacks: list | None = None, verbose: int = 0):
        import tensorflow as tf

        c = self.cfg
        tf.keras.utils.set_random_seed(c.seed)
        df = clean(df_raw)
        if len(df) < 30:
            raise ValueError(f"Datos de entrenamiento insuficientes tras la limpieza ({len(df)} filas).")

        self.cat_cols = [x for x in BASE_CAT + OPT_CAT if x in df.columns]
        self.pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ("num", StandardScaler(), NUM_COLS),
        ])
        X = self.pre.fit_transform(make_features(df, self.cat_cols)).astype("float32")
        y_raw = df["Rendimiento"].to_numpy(float)
        self.y_scaler.fit(self._fwd_y(y_raw).reshape(-1, 1))
        y = self.y_scaler.transform(self._fwd_y(y_raw).reshape(-1, 1)).ravel()

        val = self._val_mask(df)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=c.patience,
                                                restore_best_weights=True)
        model = self._build(X.shape[1])
        h = model.fit(X[~val], y[~val], validation_data=(X[val], y[val]), epochs=c.epochs,
                        batch_size=c.batch_size, callbacks=[es] + (callbacks or []), verbose=verbose)
        self.history = h.history
        self.best_epoch = int(np.argmin(h.history["val_loss"])) + 1

        pred_val = self._inv_y(model.predict(X[val], verbose=0))
        self.val_metrics = regression_metrics(y_raw[val], pred_val)
        base = df[~val].groupby("Cultivo")["Rendimiento"].mean()
        base_pred = df[val]["Cultivo"].map(base).fillna(df[~val]["Rendimiento"].mean())
        self.val_baseline = regression_metrics(y_raw[val], base_pred)
 
        if c.refit_full:  # reentrena con TODO el histórico (incluye el último año)
            tf.keras.utils.set_random_seed(c.seed)
            model = self._build(X.shape[1])
            model.fit(X, y, epochs=self.best_epoch, batch_size=c.batch_size, verbose=verbose)
        self.model = model
        self.train_df = df
        self._fit_harvest_ratio(df)
        return self

    # Razón mediana Área Cosechada / Área Sembrada por cultivo (para estimar la cosecha de 2025).
    def _fit_harvest_ratio(self, df: pd.DataFrame):
        
        self.harvest_ratio, self.harvest_ratio_global = pd.Series(dtype=float), 1.0
        if "Area_Cosechada" in df.columns:
            r = (df["Area_Cosechada"] / df["Area_Sembrada"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            r = r.dropna()
            self.harvest_ratio_global = float(r.median())
            self.harvest_ratio = r.groupby(df.loc[r.index, "Cultivo"]).median()

    # -- predicción
    def predict_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = clean(df_raw, require_target=False)
        absent = [c for c in self.cat_cols if c not in df.columns]
        if absent:
            raise ValueError(f"El CSV de test no trae las columnas usadas en entrenamiento: {absent}")
        X = self.pre.transform(make_features(df, self.cat_cols)).astype("float32")
        pred = self._inv_y(self.model.predict(X, verbose=0).ravel())

        keep = [c for c in ["Cultivo", "Ciclo", "Periodo", "Anio", "Municipio", "Subregion",
                            "Area_Sembrada"] if c in df.columns]
        out = df[keep].copy()
        out["Anio"] = out["Anio"].astype(int)
        out["Rendimiento_Predicho"] = pred.round(4)
        # Rendimiento = t por ha COSECHADA -> producción = rendimiento × área cosechada.
        # Si el CSV de test trae Area_Cosechada se usa; si no, se estima: sembrada × razón histórica del cultivo.
        est = df["Area_Sembrada"] * df["Cultivo"].map(self.harvest_ratio).fillna(self.harvest_ratio_global)
        area_c = df["Area_Cosechada"].fillna(est) if "Area_Cosechada" in df.columns else est
        out["Area_Cosechada_Usada"] = area_c.round(2).to_numpy()
        out["Produccion_Estimada_t"] = (pred * area_c.to_numpy()).round(3)
        if "Rendimiento" in df.columns and df["Rendimiento"].notna().any():
            real = df["Rendimiento"].to_numpy(float)
            err = np.abs(pred - real)
            out["Rendimiento_Real"] = real
            out["Error_Abs"] = err.round(4)
            out["Error_Pct"] = np.where(real > 0, 100 * err / np.where(real > 0, real, 1), np.nan).round(2)
        return out

# Callback de Keras que llama fn(epoca, logs) (útil para barras de progreso)
def progress_callback(fn):
    import tensorflow as tf

    class _CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            fn(epoch + 1, logs or {})

    return _CB()

# ----------------------------------------------------------------------------- pipeline
# Configuración -> entrenamiento (2021-2024) -> predicción (2025) -> DataFrame de salida
def run_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cfg: PredictionConfig | None = None, callbacks: list | None = None) -> dict:
    predictor = YieldPredictor(cfg).fit(train_df, callbacks=callbacks)
    preds = predictor.predict_dataframe(test_df)

    result = {"predictor": predictor, "predictions": preds,
                "val_metrics": predictor.val_metrics, "val_baseline": predictor.val_baseline,
                "test_metrics": None, "test_baseline": None}
    if "Rendimiento_Real" in preds.columns:
        real = preds["Rendimiento_Real"]
        result["test_metrics"] = regression_metrics(real, preds["Rendimiento_Predicho"])
        means = predictor.train_df.groupby("Cultivo")["Rendimiento"].mean()
        base = preds["Cultivo"].map(means).fillna(predictor.train_df["Rendimiento"].mean())
        result["test_baseline"] = regression_metrics(real, base)
    return result

# Guarda el CSV de predicciones. Sin ruta: output/predicciones_<fecha_hora>.csv
def save_predictions(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    path = Path(path) if path else OUTPUT_DIR / f"predicciones_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

if __name__ == "__main__":
    # los argumentos son formas de relacionar direcciones o comandos dentro de variables o alias que pueden ser llamada 
    # los registros ocurren directamente en la consola shell
    ap = argparse.ArgumentParser(description="Predicción de rendimiento agrícola en Sucre (MLP)")
    ap.add_argument("--train", required=True, type=Path, help="CSV 2021-2024")
    ap.add_argument("--test", required=True, type=Path, help="CSV 2025")
    ap.add_argument("--out", type=Path, default=None, help="por defecto: output/predicciones_<fecha>.csv")
    a = ap.parse_args()
    res = run_pipeline(read_csv(a.train), read_csv(a.test))   # read_csv: ruta -> DataFrame
    path = save_predictions(res["predictions"], a.out)
    print("Validación:", res["val_metrics"], "| baseline:", res["val_baseline"])
    print("Test 2025:", res["test_metrics"], "| baseline:", res["test_baseline"])
    print(f"Guardado en {path}")
# Modelos entrenados

En este directorio se almacenan los artefactos generados por `scripts/train_model.py`:

- `price_model.joblib` (~519 KB): `Pipeline` de scikit-learn (`ColumnTransformer` con `OneHotEncoder` sobre distrito, micro-barrio y tipología + variables numéricas) unido a un `TransformedTargetRegressor(XGBRegressor)` con transformación logarítmica (`log1p` / `expm1`).
- `model_metadata.json`: Metadatos completos de entrenamiento sobre `21.551` viviendas reales de Kaggle (`houses_Madrid.csv`), comparativa de benchmark (`Ridge`, `HistGradientBoosting`, `XGBoost`), importancias agrupadas, factor temporal `×1.35` y medianas `€/m²` por distrito y por los `126` micro-barrios.
- `comparables_index.json.gz` (~128 KB): Índice comprimido y 100 % anonimizado de testigos reales del mercado para mostrar inmuebles comparables por zona y similitud.

Para reentrenar con el dataset real de Kaggle:

```bash
python scripts/train_model.py --dataset data/raw/houses_Madrid.csv
```

Para entrenar con el dataset sintético reproducible incluido en el repo:

```bash
python scripts/train_model.py
```

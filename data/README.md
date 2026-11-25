# Datos del proyecto

Este directorio separa los datos externos originales (`raw/`) del dataset limpio listo para entrenamiento (`processed/`).

## Estructura

- `raw/`: Coloca aquí cualquier CSV descargado de Kaggle (por ejemplo, `houses_Madrid.csv`). Los archivos `.csv` de esta carpeta están ignorados en `.gitignore` para no subir datasets pesados al repositorio.
- `processed/madrid_housing_sample.csv`: Dataset sintético realista y reproducible (4.000 viviendas repartidas entre los 21 distritos oficiales de Madrid) generado con `scripts/generate_madrid_dataset.py`. Permite clonar el repo y entrenar el modelo tabular en segundos sin dependencias externas.

## Esquema esperado para entrenamiento

El script `scripts/train_model.py` acepta tanto el esquema limpio en español como columnas habituales de datasets públicos de Madrid en Kaggle (mapeadas automáticamente):

| Columna objetivo | Alias de Kaggle soportados | Descripción |
| --- | --- | --- |
| `barrio` | `district`, `neighborhood_id` | Distrito de Madrid |
| `micro_barrio` | `neighborhood_id` (extraído por regex), `neighborhood_name` | Barrio específico (ej. `Recoletos`, `Goya`, `Delicias`) |
| `tipo_vivienda` | `house_type_id`, `property_type` | `Piso`, `Ático`, `Dúplex` o `Casa o chalet` |
| `metros_cuadrados` | `sq_mt_built`, `sq_mt_useful`, `area` | Superficie construida en m² |
| `habitaciones` | `n_rooms`, `rooms` | Número de dormitorios |
| `banos` | `n_bathrooms`, `bathrooms` | Número de baños |
| `planta` | `floor` | Planta de la vivienda (`0` = bajo) |
| `ascensor` | `has_lift`, `lift` | Booleano (`0`/`1` o `True`/`False`) |
| `exterior` | `is_exterior` | Vivienda exterior (`0`/`1`) |
| `terraza` | `has_terrace` | Dispone de terraza (`0`/`1`) |
| `garaje` | `has_parking`, `is_parking_included_in_price` | Plaza de garaje incluida (`0`/`1`) |
| `aire_acondicionado` | `has_ac` | Aire acondicionado (`0`/`1`) |
| `trastero` | `has_storage_room` | Trastero (`0`/`1`) |
| `necesita_reforma` | `is_renewal_needed` | Estado a reformar en el CSV (`0`/`1`) |
| `obra_nueva` | `is_new_development` | Promoción de obra nueva (`0`/`1`) |
| `precio` | `buy_price`, `price` | Precio de venta en EUR |

## Cómo entrenar con un dataset de Kaggle

> La aplicación **ya funciona sin descargar este CSV** porque el modelo entrenado (`models/price_model.joblib`) y el índice anonimizado de comparables (`models/comparables_index.json.gz`) ya están incluidos en el repositorio.

1. Busca en Kaggle el dataset **[`Madrid real estate market`](https://www.kaggle.com/datasets?search=Madrid+real+estate+market)** y descarga el archivo `houses_Madrid.csv` (`21.742` filas y `58` columnas).
2. Guárdalo exactamente en esta ruta (queda ignorado por `.gitignore` para no subir anuncios brutos con direcciones de calle ni ficheros pesados al historial de Git):
   ```text
   data/raw/houses_Madrid.csv
   ```
3. Ejecuta:
   ```bash
   python scripts/train_model.py --dataset data/raw/houses_Madrid.csv
   ```
4. Reinicia la API de FastAPI (`uvicorn tasador.main:app --reload`) para cargar el nuevo modelo generado en `models/`.

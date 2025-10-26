# Tenbaggers Detector

Sistema modular para detectar acciones con potencial de convertirse en "tenbaggers" siguiendo la especificación 10x Hunter.

## Características

- Descarga datos OHLCV ajustados desde Yahoo! Finance o CSV propios.
- Limpieza de series temporales y construcción de universo con filtros de precio y liquidez.
- Cálculo de medias móviles (20/50/100/200), pendientes, compresión y z-score de volumen.
- Aproximación de Volume Profile (VPVR) visible con detección de POC, VAH/VAL, HVN y LVN.
- Señal de ruptura con confirmación de estructura alcista y compresión previa.
- Sistema de scoring 0-100 y salida JSON por ticker compatible con la especificación.

## Instalación

```bash
pip install -e .[dev]
```

## Uso

Desde la línea de comandos:

```bash
python -m tenbaggers_detector.cli TICKER1 TICKER2 --start 2010-01-01 --source yfinance --output resultados.json
```

### Parámetros relevantes

- `--min-price`: Precio máximo permitido (≤ 40 por defecto).
- `--min-dollar-volume`: Liquidez mínima (ADV) en USD.
- `--lookback-years`: Ventana para el Volume Profile.
- `--zscore`: Umbral del z-score de volumen para rupturas.
- `--compression`: Percentil máximo de volatilidad de 60 días.

La salida es un JSON con todos los campos solicitados (POC, VAH, LVN, medias móviles, score, notas, etc.).

## Desarrollo

Ejecutar la suite de pruebas:

```bash
pytest
```

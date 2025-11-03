# Tenbaggers Detector

Sistema modular para detectar acciones con potencial de convertirse en "tenbaggers" siguiendo la especificación 10x Hunter.

## Características

### Core Features
- Descarga datos OHLCV ajustados desde Yahoo! Finance o CSV propios.
- Limpieza de series temporales y construcción de universo con filtros de precio y liquidez.
- Cálculo de medias móviles (20/50/100/200), pendientes, compresión y z-score de volumen.
- Aproximación de Volume Profile (VPVR) visible con detección de POC, VAH/VAL, HVN y LVN.
- Señal de ruptura con confirmación de estructura alcista y compresión previa.
- Sistema de scoring 0-100 y salida JSON por ticker compatible con la especificación.

### 🆕 Enhanced Analysis Features
- **Outlier Detection**: Kolmogorov-Smirnov test para identificar tickers con comportamiento anómalo
- **Signal Quality Analysis**: Filtrado de señales redundantes y métricas de calidad
- **Robustness Validation**: Validación estadística para prevenir overfitting
- **Comprehensive Reporting**: Reportes detallados con recomendaciones accionables

### 🧠 Arquitectura orientada a estados
- **State Machine Pipeline**: `EnhancedPipeline` ahora delega cada fase (detección, outliers, filtrado, validación) a estados explícitos (`enhanced_states.py`).
- **Encapsulación del cambio**: cada estado conoce su responsabilidad y puede evolucionar sin romper el resto del flujo.
- **Polimorfismo sobre condicionales**: las transiciones reemplazan condicionales anidados, lo que permite añadir pasos sin modificar el núcleo.
- **Métricas cohesionadas**: el análisis de calidad se recalcula automáticamente al transicionar entre estados.

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

### Uso Programático con Enhanced Pipeline

```python
from tenbaggers_detector.enhanced_pipeline import EnhancedPipeline, EnhancedConfig
from tenbaggers_detector.data.sources import YFinanceSource

# Configurar pipeline con validación estadística
config = EnhancedConfig(
    enable_outlier_detection=True,      # Detectar outliers con KS test
    enable_robustness_validation=True,  # Validar robustez de la estrategia
    enable_signal_filtering=True,       # Filtrar señales redundantes
    verbose=True,                        # Mostrar reportes detallados
)

# Ejecutar análisis
source = YFinanceSource()
pipeline = EnhancedPipeline(source, config)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
results = pipeline.run(tickers, start='2020-01-01', end='2023-12-31')

# Obtener reporte completo
print(pipeline.get_analysis_report())

# Ver tickers excluidos como outliers
excluded = pipeline.get_excluded_universe()
print(f"Tickers excluidos: {excluded}")
```

Ver [documentación completa de análisis de outliers](docs/OUTLIER_ANALYSIS.md) para más detalles.

## Desarrollo

Ejecutar la suite de pruebas:

```bash
pytest
```

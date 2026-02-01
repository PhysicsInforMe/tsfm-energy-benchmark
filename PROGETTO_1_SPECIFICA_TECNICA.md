# PROGETTO 1: Time Series Foundation Model Benchmark per Energy Load Forecasting

## Documento di Specifica Tecnica per Claude Code

**Autore**: Luigi Simeone  
**Repository**: https://github.com/PhysicsInforMe/scientific-prototypes  
**Timeline**: 2 settimane (Febbraio 2025)  
**Ambiente**: Google Colab Free (T4 GPU, ~12GB VRAM)

---

## 1. OBIETTIVO DEL PROGETTO

### 1.1 Obiettivo Principale
Creare un benchmark rigoroso che confronti i principali Time Series Foundation Models (TSFM) su task di energy load forecasting, dimostrando competenze in:
- Modellazione di serie temporali con approcci state-of-the-art
- Valutazione rigorosa con metriche standard del settore
- Applicazione pratica a problemi energetici reali

### 1.2 Valore per la Comunità
- Benchmark riproducibile su dati pubblici
- Confronto diretto tra TSFM e baseline statistiche
- Analisi di robustezza su diversi orizzonti di previsione
- Codice production-ready con best practice

### 1.3 Narrativa
Questo progetto dimostra la capacità di applicare modelli fondazionali a problemi energetici reali, evidenziando come le stesse tecniche matematiche (tokenizzazione di serie temporali, attention mechanism) possano essere trasferite a domini diversi.

---

## 2. DATASET

### 2.1 Dataset Primario: ERCOT Hourly Load Data

**Fonte**: https://www.ercot.com/gridinfo/load/load_hist

**Descrizione**: Dati orari di carico elettrico della rete ERCOT (Electric Reliability Council of Texas), una delle principali reti elettriche indipendenti negli USA.

**Caratteristiche**:
- Frequenza: Oraria (8760 osservazioni/anno)
- Periodo disponibile: 2010-2024 (15 anni)
- Granularità: Carico totale ERCOT + suddivisione per zone
- Formato: ZIP contenente file Excel/CSV

**Perché ERCOT**:
- Dati pubblici e gratuiti (no API key richiesta)
- Rete elettrica con pattern stagionali chiari (estate calda in Texas)
- Sufficiente storia per test robusti
- Ben documentato nella letteratura

### 2.2 Download e Preprocessing

```python
# URL dei dati ERCOT (da scaricare manualmente o con requests)
ERCOT_URLS = {
    "2024": "https://www.ercot.com/files/docs/2025/01/10/Native_Load_2024.zip",
    "2023": "https://www.ercot.com/files/docs/2024/01/08/Native_Load_2023.zip",
    "2022": "https://www.ercot.com/files/docs/2023/01/31/Native_Load_2022.zip",
    "2021": "https://www.ercot.com/files/docs/2022/01/07/Native_Load_2021.zip",
    "2020": "https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip",
}

# Colonne da estrarre
TARGET_COLUMN = "ERCOT"  # Carico totale della rete
TIMESTAMP_COLUMN = "Hour_Ending"
```

### 2.3 Split dei Dati

```
Training:    2020-01-01 → 2022-12-31 (3 anni, ~26,280 ore)
Validation:  2023-01-01 → 2023-06-30 (6 mesi, ~4,380 ore)  
Test:        2023-07-01 → 2024-06-30 (1 anno, ~8,760 ore)
```

**Razionale**: 
- Test set include sia estate che inverno per testare generalizzazione stagionale
- Training set abbastanza lungo per baseline statistiche
- Validation per hyperparameter tuning se necessario

---

## 3. MODELLI DA CONFRONTARE

### 3.1 Time Series Foundation Models (Zero-Shot)

#### A) Chronos-Bolt (Base) - PRIORITÀ ALTA
- **Modello**: `amazon/chronos-bolt-base` (205M parametri)
- **Architettura**: T5 encoder-decoder con patch-based input
- **Caratteristiche**: 250x più veloce di Chronos originale, memory-efficient
- **VRAM richiesta**: ~4-6 GB (compatibile con Colab free)
- **Libreria**: `chronos-forecasting`

```python
# Installazione
!pip install chronos-forecasting

# Utilizzo
from chronos import ChronosBoltPipeline

pipeline = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda"
)
```

#### B) Chronos-2 - PRIORITÀ ALTA
- **Modello**: `amazon/chronos-2` (120M parametri)
- **Architettura**: Encoder-only con group attention
- **Caratteristiche**: Supporta covariate, multivariate (opzionale per questo progetto)
- **VRAM richiesta**: ~4-6 GB
- **Novità**: Rilasciato ottobre 2024, stato dell'arte su fev-bench

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)
```

#### C) Lag-Llama - PRIORITÀ MEDIA
- **Modello**: `time-series-foundation-models/Lag-Llama` (~7M parametri)
- **Architettura**: Decoder-only transformer con lag features
- **Caratteristiche**: Primo TSFM open-source, forecasting probabilistico
- **VRAM richiesta**: ~2-4 GB
- **Libreria**: Richiede clone del repo + GluonTS

```python
# Installazione
!git clone https://github.com/time-series-foundation-models/lag-llama/
!pip install -r lag-llama/requirements.txt
!huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./lag-llama
```

### 3.2 Baseline Statistiche (per confronto)

#### D) Seasonal Naive
- Previsione = valore stesso orario della settimana precedente
- Baseline minima per serie stagionali

#### E) ARIMA/SARIMA
- Modello classico con componenti stagionali
- Implementazione: `statsmodels` o `pmdarima`

#### F) Prophet (opzionale)
- Modello Facebook per serie con stagionalità multipla
- Buon baseline per confronto con TSFM

### 3.3 Deep Learning Baseline (se tempo permette)

#### G) DeepAR (opzionale)
- RNN-based probabilistic forecasting
- Implementazione: GluonTS
- Serve come ponte tra statistiche e TSFM

---

## 4. METRICHE DI VALUTAZIONE

### 4.1 Metriche Principali

#### A) MASE (Mean Absolute Scaled Error) - PRIORITÀ ALTA
```python
def mase(y_true, y_pred, y_train, seasonality=24):
    """
    MASE scala l'errore rispetto a un naive stagionale.
    MASE < 1 significa che il modello batte il naive.
    """
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d
```

#### B) CRPS (Continuous Ranked Probability Score) - per forecast probabilistici
```python
def crps(y_true, forecast_samples):
    """
    CRPS misura la qualità delle distribuzioni predette.
    Più basso è meglio.
    """
    from properscoring import crps_ensemble
    return crps_ensemble(y_true, forecast_samples).mean()
```

#### C) WQL (Weighted Quantile Loss) - PRIORITÀ ALTA
```python
def weighted_quantile_loss(y_true, y_pred_quantiles, quantiles=[0.1, 0.5, 0.9]):
    """
    WQL è la metrica standard per forecast probabilistici.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred_quantiles[:, i]
        losses.append(np.maximum(q * errors, (q - 1) * errors).mean())
    return np.mean(losses)
```

#### D) MAE e RMSE - metriche point forecast
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

### 4.2 Orizzonti di Previsione da Testare

```python
PREDICTION_HORIZONS = [
    24,    # 1 giorno (day-ahead forecasting, standard nel mercato energetico)
    168,   # 1 settimana
    720,   # ~1 mese (30 giorni)
]
```

**Razionale**: Il day-ahead (24h) è lo standard industriale per il mercato elettrico. Testare orizzonti più lunghi mostra robustezza.

### 4.3 Context Length

```python
CONTEXT_LENGTHS = [
    512,   # ~21 giorni (default per molti TSFM)
    1024,  # ~42 giorni (test di memoria più lunga)
]
```

---

## 5. STRUTTURA DEL REPOSITORY

```
energy-load-forecasting-benchmark/
│
├── README.md                          # Documentazione principale (800-1200 parole)
├── pyproject.toml                     # Configurazione pacchetto Python moderno
├── Makefile                           # Comandi: make setup, make benchmark, make demo
├── LICENSE                            # MIT License
│
├── configs/
│   └── benchmark_config.yaml          # Configurazione esperimenti
│
├── src/
│   └── energy_benchmark/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ercot_loader.py        # Download e preprocessing ERCOT
│       │   └── preprocessing.py       # Normalizzazione, split, etc.
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                # Classe astratta ForecastModel
│       │   ├── chronos_bolt.py        # Wrapper Chronos-Bolt
│       │   ├── chronos2.py            # Wrapper Chronos-2
│       │   ├── lag_llama.py           # Wrapper Lag-Llama
│       │   ├── statistical.py         # ARIMA, Seasonal Naive
│       │   └── prophet_model.py       # Prophet (opzionale)
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # MASE, CRPS, WQL, MAE, RMSE
│       │   └── benchmark.py           # Classe per eseguire benchmark
│       │
│       └── visualization/
│           ├── __init__.py
│           └── plots.py               # Grafici forecast, confronti, etc.
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA del dataset ERCOT
│   ├── 02_model_comparison.ipynb      # Notebook principale con tutti i confronti
│   └── 03_results_analysis.ipynb      # Analisi approfondita dei risultati
│
├── scripts/
│   ├── download_data.py               # Script per scaricare dati ERCOT
│   └── run_benchmark.py               # CLI per eseguire benchmark completo
│
├── results/
│   ├── tables/                        # Tabelle CSV con risultati
│   └── figures/                       # Grafici PNG/PDF
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_models.py
│
├── demo/
│   └── streamlit_app.py               # Demo interattiva (opzionale)
│
└── .github/
    └── workflows/
        └── ci.yml                     # GitHub Actions per CI
```

---

## 6. SPECIFICHE DI IMPLEMENTAZIONE

### 6.1 Classe Base per i Modelli

```python
# src/energy_benchmark/models/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """
    Classe base astratta per tutti i modelli di forecasting.
    Garantisce un'interfaccia uniforme per il benchmark.
    """
    
    def __init__(self, name: str, requires_gpu: bool = False):
        """
        Inizializza il modello.
        
        Args:
            name: Nome identificativo del modello
            requires_gpu: Se True, il modello richiede GPU
        """
        self.name = name
        self.requires_gpu = requires_gpu
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, train_data: pd.Series) -> "ForecastModel":
        """
        Addestra il modello (se necessario).
        Per modelli zero-shot, questo metodo può essere no-op.
        
        Args:
            train_data: Serie temporale di training con DatetimeIndex
            
        Returns:
            self per method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        context: pd.Series, 
        prediction_length: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Genera previsioni.
        
        Args:
            context: Serie temporale di contesto (storia recente)
            prediction_length: Numero di step da prevedere
            num_samples: Numero di sample per forecast probabilistico
            
        Returns:
            Tuple di (point_forecast, forecast_samples)
            - point_forecast: shape (prediction_length,)
            - forecast_samples: shape (num_samples, prediction_length) o None
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
```

### 6.2 Wrapper Chronos-Bolt

```python
# src/energy_benchmark/models/chronos_bolt.py

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from .base import ForecastModel


class ChronosBoltModel(ForecastModel):
    """
    Wrapper per Chronos-Bolt foundation model.
    
    Chronos-Bolt utilizza un'architettura T5 encoder-decoder con input patch-based.
    È 250x più veloce del Chronos originale e produce forecast probabilistici
    direttamente come quantili.
    """
    
    def __init__(
        self, 
        model_size: str = "base",
        device: str = "cuda"
    ):
        """
        Inizializza Chronos-Bolt.
        
        Args:
            model_size: Dimensione del modello ('tiny', 'mini', 'small', 'base')
            device: Device per inference ('cuda' o 'cpu')
        """
        super().__init__(
            name=f"Chronos-Bolt-{model_size.capitalize()}", 
            requires_gpu=(device == "cuda")
        )
        self.model_size = model_size
        self.device = device
        self.pipeline = None
        
    def fit(self, train_data: pd.Series) -> "ChronosBoltModel":
        """
        Chronos-Bolt è zero-shot, quindi fit() carica solo il modello.
        """
        # Import lazy per evitare errori se chronos non è installato
        from chronos import ChronosBoltPipeline
        
        model_name = f"amazon/chronos-bolt-{self.model_size}"
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=self.device
        )
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        context: pd.Series, 
        prediction_length: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Genera forecast probabilistico con Chronos-Bolt.
        
        Chronos-Bolt produce direttamente quantili, non samples.
        Per uniformità, generiamo samples approssimati dalla distribuzione.
        """
        if not self._is_fitted:
            raise RuntimeError("Chiamare fit() prima di predict()")
        
        # Converti in tensor
        context_tensor = torch.tensor(context.values, dtype=torch.float32)
        
        # Genera forecast (quantili di default: 0.1, 0.2, ..., 0.9)
        quantile_levels = [0.1, 0.5, 0.9]
        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels
        )
        
        # Estrai point forecast (mediana = quantile 0.5)
        point_forecast = forecast[:, 1].numpy()  # indice 1 = 0.5 quantile
        
        # Per compatibilità, generiamo samples dalla distribuzione approssimata
        # usando i quantili come guide (distribuzione normale troncata)
        q10 = forecast[:, 0].numpy()
        q90 = forecast[:, 2].numpy()
        
        # Approssimazione: std ≈ (q90 - q10) / 2.56 per normale
        std_approx = (q90 - q10) / 2.56
        samples = np.random.normal(
            loc=point_forecast, 
            scale=std_approx, 
            size=(num_samples, prediction_length)
        )
        
        return point_forecast, samples
```

### 6.3 Configurazione YAML

```yaml
# configs/benchmark_config.yaml

# Configurazione Dataset
data:
  source: "ercot"
  years: [2020, 2021, 2022, 2023, 2024]
  target_column: "ERCOT"
  frequency: "H"  # Hourly
  
  split:
    train_end: "2022-12-31"
    val_end: "2023-06-30"
    test_end: "2024-06-30"

# Configurazione Modelli
models:
  chronos_bolt:
    enabled: true
    size: "base"
    device: "cuda"
    
  chronos_2:
    enabled: true
    device: "cuda"
    
  lag_llama:
    enabled: true
    context_length: 512
    
  seasonal_naive:
    enabled: true
    seasonality: 168  # settimanale
    
  arima:
    enabled: true
    order: [2, 1, 2]
    seasonal_order: [1, 1, 1, 24]

# Configurazione Benchmark
benchmark:
  prediction_horizons: [24, 168, 720]
  context_lengths: [512, 1024]
  num_samples: 100
  
  # Rolling window evaluation
  rolling:
    enabled: true
    step_size: 24  # Valuta ogni 24 ore
    num_windows: 30  # 30 finestre di test

# Configurazione Metriche
metrics:
  - mase
  - crps
  - wql
  - mae
  - rmse

# Output
output:
  results_dir: "results"
  save_predictions: true
  save_figures: true
```

### 6.4 Script di Benchmark

```python
# scripts/run_benchmark.py

"""
Script principale per eseguire il benchmark completo.

Utilizzo:
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml
    
Oppure con Makefile:
    make benchmark
"""

import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from energy_benchmark.data import ERCOTLoader
from energy_benchmark.models import (
    ChronosBoltModel, 
    Chronos2Model,
    LagLlamaModel,
    SeasonalNaiveModel,
    ARIMAModel
)
from energy_benchmark.evaluation import BenchmarkRunner, MetricsCalculator
from energy_benchmark.visualization import plot_comparison, plot_forecasts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """
    Esegue il benchmark completo.
    
    Args:
        config_path: Percorso al file di configurazione YAML
    """
    # Carica configurazione
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configurazione caricata da {config_path}")
    
    # 1. Carica dati
    logger.info("Caricamento dati ERCOT...")
    loader = ERCOTLoader(years=config['data']['years'])
    data = loader.load()
    
    train, val, test = loader.split(
        data,
        train_end=config['data']['split']['train_end'],
        val_end=config['data']['split']['val_end']
    )
    
    logger.info(f"Train: {len(train)} ore, Val: {len(val)} ore, Test: {len(test)} ore")
    
    # 2. Inizializza modelli
    models = []
    model_config = config['models']
    
    if model_config['chronos_bolt']['enabled']:
        models.append(ChronosBoltModel(
            model_size=model_config['chronos_bolt']['size'],
            device=model_config['chronos_bolt']['device']
        ))
    
    if model_config['chronos_2']['enabled']:
        models.append(Chronos2Model(
            device=model_config['chronos_2']['device']
        ))
    
    if model_config['lag_llama']['enabled']:
        models.append(LagLlamaModel(
            context_length=model_config['lag_llama']['context_length']
        ))
    
    if model_config['seasonal_naive']['enabled']:
        models.append(SeasonalNaiveModel(
            seasonality=model_config['seasonal_naive']['seasonality']
        ))
    
    if model_config['arima']['enabled']:
        models.append(ARIMAModel(
            order=tuple(model_config['arima']['order']),
            seasonal_order=tuple(model_config['arima']['seasonal_order'])
        ))
    
    logger.info(f"Modelli inizializzati: {[m.name for m in models]}")
    
    # 3. Esegui benchmark
    runner = BenchmarkRunner(
        models=models,
        metrics=config['metrics'],
        prediction_horizons=config['benchmark']['prediction_horizons'],
        context_lengths=config['benchmark']['context_lengths'],
        num_samples=config['benchmark']['num_samples']
    )
    
    results = runner.run(train, test, config['benchmark']['rolling'])
    
    # 4. Salva risultati
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tabella riassuntiva
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "benchmark_results.csv", index=False)
    logger.info(f"Risultati salvati in {output_dir / 'benchmark_results.csv'}")
    
    # Figure
    if config['output']['save_figures']:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        plot_comparison(results_df, save_path=fig_dir / "model_comparison.png")
        plot_forecasts(test, results['predictions'], save_path=fig_dir / "forecasts.png")
    
    logger.info("Benchmark completato!")
    
    # Stampa riassunto
    print("\n" + "="*60)
    print("RISULTATI BENCHMARK")
    print("="*60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run energy forecasting benchmark")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/benchmark_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)
```

---

## 7. README.md TEMPLATE

```markdown
# ⚡ Energy Load Forecasting with Time Series Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhysicsInforMe/scientific-prototypes/blob/main/energy-load-forecasting-benchmark/notebooks/02_model_comparison.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A rigorous benchmark comparing **Time Series Foundation Models** (Chronos-2, Chronos-Bolt, Lag-Llama) against classical methods for **day-ahead electricity load forecasting** on ERCOT data.

![Benchmark Results](results/figures/model_comparison.png)

## 🎯 Key Findings

| Model | MASE (24h) | MASE (168h) | Inference Time |
|-------|-----------|-------------|----------------|
| Chronos-Bolt (Base) | **0.82** | **0.91** | 0.3s |
| Chronos-2 | 0.85 | 0.93 | 0.5s |
| Lag-Llama | 0.89 | 0.97 | 1.2s |
| SARIMA | 0.95 | 1.12 | 45s |
| Seasonal Naive | 1.00 | 1.00 | <0.1s |

**Main insight**: Zero-shot foundation models outperform traditional methods without any task-specific training, demonstrating remarkable transfer learning capabilities.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/PhysicsInforMe/scientific-prototypes.git
cd scientific-prototypes/energy-load-forecasting-benchmark
pip install -e .

# Run benchmark
make benchmark

# Or use the interactive notebook
make notebook
```

## 📊 Dataset

We use [ERCOT Hourly Load Data](https://www.ercot.com/gridinfo/load/load_hist) (2020-2024), one of the largest publicly available electricity datasets:
- **87,600 hourly observations** across 5 years
- Clear seasonal patterns (Texas summer peaks)
- Real-world operational relevance

## 🧠 Models Compared

### Foundation Models (Zero-Shot)
- **Chronos-2** (Amazon, 2024): Latest TSFM with covariate support
- **Chronos-Bolt**: 250x faster variant with direct quantile forecasting  
- **Lag-Llama**: First open-source TSFM with probabilistic outputs

### Baselines
- SARIMA: Classical statistical approach
- Seasonal Naive: Persistence baseline (same hour last week)

## 📈 Metrics

- **MASE**: Mean Absolute Scaled Error (scale-free comparison)
- **CRPS**: Continuous Ranked Probability Score (probabilistic accuracy)
- **WQL**: Weighted Quantile Loss at [0.1, 0.5, 0.9]

## 🔬 Methodology

We evaluate all models using **rolling window validation**:
1. Context window: 512 hours (~21 days)
2. Forecast horizons: 24h (day-ahead), 168h (week-ahead), 720h (month-ahead)
3. 30 test windows across 2023-2024

## 📁 Repository Structure

```
├── src/energy_benchmark/    # Core library
├── notebooks/               # Jupyter notebooks
├── scripts/                 # CLI tools
├── configs/                 # YAML configurations
└── results/                 # Output tables and figures
```

## 🙏 Citation

If you find this benchmark useful, please cite:

```bibtex
@software{simeone2025energy,
  author = {Simeone, Luigi},
  title = {Energy Load Forecasting with Time Series Foundation Models},
  year = {2025},
  url = {https://github.com/PhysicsInforMe/scientific-prototypes}
}
```

## 📚 References

- Ansari et al. (2024). [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- Rasul et al. (2024). [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)

## 📝 License

MIT License - feel free to use and adapt for your research!
```

---

## 8. TIMELINE DETTAGLIATA

### Settimana 1 (Giorni 1-7)

| Giorno | Task | Output |
|--------|------|--------|
| 1-2 | Setup repository, pyproject.toml, struttura cartelle | Scheletro repo |
| 2-3 | Implementare ERCOTLoader + preprocessing | `ercot_loader.py` funzionante |
| 3-4 | Implementare wrapper Chronos-Bolt e Chronos-2 | Modelli testati su sample |
| 5-6 | Implementare Lag-Llama wrapper | Tutti i TSFM funzionanti |
| 7 | Implementare baseline (Naive, ARIMA) | Baseline pronte |

### Settimana 2 (Giorni 8-14)

| Giorno | Task | Output |
|--------|------|--------|
| 8-9 | Implementare metriche e BenchmarkRunner | `evaluation/` completo |
| 10 | Eseguire benchmark completo | CSV risultati |
| 11 | Creare visualizzazioni | Figure in `results/` |
| 12 | Scrivere notebook 01 e 02 | Notebook documentati |
| 13 | Scrivere README, aggiungere badge Colab | README completo |
| 14 | Review finale, push, test CI | Repo pubblicata |

---

## 9. REQUISITI COLAB

### 9.1 Setup Cell (inizio notebook)

```python
# Installa dipendenze
!pip install -q chronos-forecasting gluonts torch pandas numpy matplotlib seaborn scikit-learn pmdarima properscoring pyyaml tqdm

# Clona Lag-Llama
!git clone https://github.com/time-series-foundation-models/lag-llama.git
!pip install -q -r lag-llama/requirements.txt
!huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./lag-llama

# Verifica GPU
import torch
print(f"GPU disponibile: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 9.2 Stima Tempi di Esecuzione

| Operazione | Tempo stimato (T4 GPU) |
|------------|------------------------|
| Download dati ERCOT | 2-3 minuti |
| Load Chronos-Bolt | 30 secondi |
| Load Chronos-2 | 30 secondi |
| Load Lag-Llama | 45 secondi |
| Benchmark completo (30 windows) | 15-20 minuti |
| Generazione figure | 1 minuto |

**Tempo totale stimato**: ~25-30 minuti per esecuzione completa

---

## 10. CHECKLIST FINALE

### Codice
- [ ] Tutti i file hanno docstring e type hints
- [ ] Codice formattato con Black
- [ ] Linting passato con Ruff
- [ ] Test unitari per data loader e metriche
- [ ] Nessun hardcoded path (usa Path e config)

### Documentazione
- [ ] README con badge, risultati, quick start
- [ ] Notebook commentati e eseguibili end-to-end
- [ ] Config YAML documentato
- [ ] requirements.txt / pyproject.toml completo

### Risultati
- [ ] Tabella comparativa modelli
- [ ] Grafici: confronto metriche, forecast examples
- [ ] Analisi per orizzonte temporale

### Pubblicazione
- [ ] Badge "Open in Colab" funzionante
- [ ] License MIT
- [ ] .gitignore appropriato
- [ ] GitHub Actions CI configurato

---

## 11. NOTE PER CLAUDE CODE

### Priorità di Implementazione
1. **Alta**: ERCOTLoader, ChronosBoltModel, metriche (MASE, WQL)
2. **Media**: Chronos2Model, LagLlamaModel, BenchmarkRunner
3. **Bassa**: ARIMA, Prophet, Streamlit demo

### Convenzioni da Seguire
- Python 3.10+ con type hints ovunque
- Docstring in stile Google
- Commenti in italiano per spiegazioni, inglese per codice
- Nomi variabili in inglese, snake_case
- Classi in PascalCase

### Gestione Errori
- Usare logging invece di print per debug
- Gestire gracefully il caso GPU non disponibile (fallback CPU)
- Timeout per download dati ERCOT

### Testing
- Usare pytest con fixtures per dati mock
- Test di integrazione nel notebook 02
- Coverage target: >70%

---

**Fine documento di specifica. Questo file può essere condiviso direttamente con Claude Code per iniziare l'implementazione.**

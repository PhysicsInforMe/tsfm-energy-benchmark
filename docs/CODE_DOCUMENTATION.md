# Documentazione Completa del Codice

**Progetto**: Energy Load Forecasting Benchmark
**Autore**: Luigi Simeone
**Versione**: 0.1.0

---

## Indice

1. [Architettura Generale](#1-architettura-generale)
2. [Modulo `data` — Caricamento e Preprocessing](#2-modulo-data)
   - 2.1 [ERCOTLoader](#21-ercotloader)
   - 2.2 [Preprocessing](#22-preprocessing)
3. [Modulo `models` — Wrapper dei Modelli](#3-modulo-models)
   - 3.1 [ForecastModel (classe base)](#31-forecastmodel-classe-base)
   - 3.2 [SeasonalNaiveModel](#32-seasonalnaivemodel)
   - 3.3 [ARIMAModel](#33-arimamodel)
   - 3.4 [ChronosBoltModel](#34-chronosboltmodel)
   - 3.5 [Chronos2Model](#35-chronos2model)
   - 3.6 [LagLlamaModel](#36-lagllamamodel)
   - 3.7 [ProphetModel](#37-prophetmodel)
4. [Modulo `evaluation` — Metriche e Benchmark Runner](#4-modulo-evaluation)
   - 4.1 [Metriche](#41-metriche)
   - 4.2 [BenchmarkRunner](#42-benchmarkrunner)
5. [Modulo `visualization` — Grafici](#5-modulo-visualization)
6. [Scripts](#6-scripts)
   - 6.1 [download_data.py](#61-download_datapy)
   - 6.2 [run_benchmark.py](#62-run_benchmarkpy)
7. [Demo Streamlit](#7-demo-streamlit)
8. [Configurazione YAML](#8-configurazione-yaml)
9. [Testing](#9-testing)
10. [Scelte Progettuali Trasversali](#10-scelte-progettuali-trasversali)

---

## 1. Architettura Generale

```
src/energy_benchmark/
    data/               Caricamento dati ERCOT + pulizia serie
    models/             Wrapper uniformi per ogni modello (ABC + 6 implementazioni)
    evaluation/         Metriche di errore + orchestratore del benchmark
    visualization/      Funzioni di plotting per confronti e analisi
scripts/                CLI per download dati e lancio benchmark
demo/                   App Streamlit interattiva
configs/                Configurazione YAML per esperimenti riproducibili
tests/                  35 test unitari (pytest)
```

**Principio guida**: ogni modello, non importa quanto diverso internamente, espone la stessa interfaccia `fit()` / `predict()`. Questo permette al `BenchmarkRunner` di iterare su una lista eterogenea di modelli senza conoscerne l'implementazione.

**Flusso dei dati**:

```
EIA Open Data API → ERCOTLoader → pd.Series → preprocess_series()
                                       ↓
                                split (train/val/test)
                                       ↓
                      ForecastModel.fit(train) → .predict(context)
                                       ↓
                         BenchmarkRunner (rolling windows)
                                       ↓
                      BenchmarkResults → DataFrame CSV + Figures
```

---

## 2. Modulo `data`

**File**: `src/energy_benchmark/data/`

### 2.1 ERCOTLoader

**File**: `ercot_loader.py`
**Scopo**: Scaricare, parsare e cachare i dati orari di domanda elettrica dalla rete ERCOT del Texas, tramite la EIA Open Data API v2.

#### Costanti di modulo

```python
EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
EIA_PAGE_SIZE = 5000
MIN_YEAR = 2015
TARGET_COLUMN = "ERCOT"
```

**Cosa**: `EIA_API_URL` e' l'endpoint dell'API EIA per i dati regionali di domanda oraria. `EIA_PAGE_SIZE` e' il numero massimo di righe per richiesta API (limite imposto dall'EIA). `MIN_YEAR` e' il primo anno con dati RTO disponibili. `TARGET_COLUMN` e' il nome assegnato alla colonna di output.

**Perche' cosi'**: L'EIA (U.S. Energy Information Administration) fornisce un'API pubblica e gratuita per accedere ai dati energetici, inclusa la domanda oraria di ERCOT. Questo approccio sostituisce il download diretto di file ZIP dal sito ERCOT, che blocca le richieste automatizzate con un WAF (Web Application Firewall). L'API EIA e' stabile, documentata e accessibile con una chiave `DEMO_KEY` (rate-limited) o con una chiave gratuita registrata.

---

#### `class ERCOTLoader`

```python
def __init__(self, years, data_dir, target_column, api_key, timeout)
```

**Cosa fa**: Inizializza il loader validando che gli anni richiesti siano >= `MIN_YEAR` (2015).

**Come lo fa**: Controlla che nessun anno sia precedente al 2015 e solleva `ValueError` se necessario. La chiave API viene cercata in ordine: parametro esplicito, variabile d'ambiente `EIA_API_KEY`, fallback a `"DEMO_KEY"`.

**Perche' cosi'**: Fail-fast: meglio fallire alla costruzione che durante una sequenza di chiamate API. La chiave `DEMO_KEY` funziona immediatamente senza registrazione (limite ~30 richieste/ora). Per uso intensivo, basta impostare la variabile d'ambiente `EIA_API_KEY` con una chiave gratuita ottenuta da https://www.eia.gov/opendata/register.php.

---

#### `ERCOTLoader.load(force_download=False) -> pd.Series`

**Cosa fa**: Restituisce l'intera serie oraria di carico in MW, combinando tutti gli anni richiesti.

**Come lo fa**:
1. Per ogni anno, chiama `_load_year()` che restituisce un DataFrame.
2. Concatena i DataFrame e ordina per indice temporale.
3. Rimuove timestamp duplicati (possibili durante le transizioni DST).
4. Estrae la colonna target e la rinomina `"load_mw"`.

**Perche' cosi'**: La concatenazione anno per anno permette il caching granulare — se un anno e' gia' stato scaricato dall'API, non viene riscaricato. La deduplicazione dei timestamp gestisce eventuali sovrapposizioni ai confini tra anni.

---

#### `ERCOTLoader.split(series, train_end, val_end) -> (train, val, test)`

**Cosa fa**: Divide la serie in tre segmenti temporali non sovrapposti.

**Come lo fa**: Usa confronti booleani sull'indice (`series.index <= Timestamp`). Il train include tutto fino a `train_end` incluso, il val va da `train_end` escluso a `val_end` incluso, il test da `val_end` escluso in poi.

**Perche' cosi'**: Inizialmente usavamo `series.loc[:]` con slicing, ma questo causava overlap ai boundary perche' `loc` include entrambi gli estremi. I confronti booleani espliciti garantiscono che `len(train) + len(val) + len(test) == len(series)` senza perdite ne' duplicazioni — verificato dai test unitari.

I default (`train_end="2022-12-31"`, `val_end="2023-06-30"`) danno:
- Train: 3 anni (~26k ore) — abbastanza per baseline statistiche
- Val: 6 mesi — per tuning opzionale
- Test: 1 anno — copre sia estate che inverno per testare la generalizzazione stagionale

---

#### `ERCOTLoader._load_year(year, force_download) -> pd.DataFrame`

**Cosa fa**: Carica i dati di un singolo anno, interrogando l'API EIA se non sono gia' in cache.

**Come lo fa**:
1. Cerca un file Parquet locale (`ercot_{year}.parquet`).
2. Se esiste e `force_download=False`, lo legge direttamente.
3. Altrimenti chiama `_fetch_year()` e salva il risultato in Parquet.

**Perche' cosi'**: Parquet e' compatto e velocissimo da leggere. Dopo il primo fetch dall'API, i caricamenti successivi sono istantanei. Questo e' critico sia nei notebook Colab (dove si riesegue spesso la cella) sia per evitare di esaurire il rate limit dell'API con `DEMO_KEY`.

---

#### `ERCOTLoader._fetch_year(year) -> pd.DataFrame`

**Cosa fa**: Scarica un anno intero di dati di domanda oraria ERCOT dall'API EIA, gestendo la paginazione automaticamente.

**Come lo fa**:
1. Costruisce i parametri della query: `respondent=ERCO`, `type=D` (Demand), intervallo `start/end` per l'anno.
2. Effettua richieste paginate (max 5000 righe per pagina) finche' non ha tutti i dati.
3. Converte il JSON di risposta in DataFrame con DatetimeIndex.
4. Converte la colonna `value` (stringa) in numerico, rinominandola in `ERCOT`.
5. Rimuove NaN e ordina per timestamp.

**Perche' cosi'**: L'API EIA restituisce al massimo 5000 righe per richiesta, ma un anno ha ~8760 ore. La paginazione con `offset` e' necessaria per ottenere tutti i dati. I valori sono restituiti come stringhe dall'API, quindi la conversione con `pd.to_numeric(errors="coerce")` gestisce eventuali valori non numerici in modo robusto. Se l'API non restituisce dati, viene sollevato un `RuntimeError` informativo.

---

### 2.2 Preprocessing

**File**: `preprocessing.py`

#### `preprocess_series(series, fill_method="linear", clip_std=5.0) -> pd.Series`

**Cosa fa**: Pulisce la serie riempiendo i NaN e rimuovendo outlier estremi.

**Come lo fa**:
1. **Interpolazione**: `fill_method="linear"` usa interpolazione temporale (tiene conto degli intervalli irregolari). Alternative: `"ffill"` (propagazione forward) e `"zero"`.
2. **Backfill**: Dopo l'interpolazione, eventuali NaN iniziali vengono riempiti con backfill.
3. **Clipping**: Valori oltre `clip_std` deviazioni standard dalla media vengono troncati.

**Perche' cosi'**: L'interpolazione lineare e' la scelta migliore per dati orari di carico perche' il carico varia in modo smooth. Il clipping a 5 sigma rimuove errori di misura evidenti (es. valori negativi o spike a 10x la media) senza toccare i picchi reali. Il default di 5.0 e' conservativo — in pratica elimina solo valori fisicamente impossibili.

---

#### `create_splits(series, train_end, val_end) -> (train, val, test)`

**Cosa fa**: Divide la serie in tre parti per data — identica alla logica di `ERCOTLoader.split()`.

**Perche' esiste come funzione separata**: Fornisce la stessa funzionalita' come funzione standalone, utilizzabile senza istanziare un `ERCOTLoader`. Utile quando i dati provengono da altra fonte.

---

#### `normalize(series, method, params=None) -> (pd.Series, dict)`

**Cosa fa**: Normalizza la serie e restituisce i parametri per l'operazione inversa.

**Come lo fa**:
- `"standard"`: z-score normalizzazione `(x - mean) / std`
- `"minmax"`: scala a [0, 1] con `(x - min) / (max - min)`

Se `params` e' fornito, li usa invece di calcolarli. Questo e' fondamentale per normalizzare il test set con i parametri del train set.

**Perche' cosi'**: La normalizzazione e' opzionale in questo benchmark (i TSFM la gestiscono internamente), ma e' disponibile per future estensioni (es. DeepAR). Restituire un dizionario `params` permette la serializzazione e la riapplicazione.

---

#### `denormalize(series, params) -> pd.Series`

**Cosa fa**: Inverte la normalizzazione.

**Come lo fa**: Applica la trasformazione inversa in base a `params["method"]`.

**Perche' cosi'**: Necessario per riportare le previsioni nella scala originale (MW) dopo aver lavorato in spazio normalizzato.

---

## 3. Modulo `models`

**File**: `src/energy_benchmark/models/`

### 3.1 ForecastModel (classe base)

**File**: `base.py`

```python
class ForecastModel(ABC)
```

**Cosa fa**: Definisce il contratto che ogni modello deve rispettare.

**Come lo fa**: Usa `ABC` (Abstract Base Class) con due metodi astratti:
- `fit(train_data) -> self`: Inizializza/addestra il modello.
- `predict(context, prediction_length, num_samples) -> (point, samples)`: Genera previsioni.

**Attributi**:
- `name: str` — Identificativo leggibile (usato nelle tabelle e nei grafici).
- `requires_gpu: bool` — Segnala se il modello necessita di CUDA.
- `_is_fitted: bool` — Flag interno che impedisce di chiamare `predict()` prima di `fit()`.

**Perche' cosi'**:

1. **Interfaccia uniforme**: Il `BenchmarkRunner` non ha bisogno di `if/else` per ogni modello. Aggiungere un nuovo modello richiede solo implementare `fit()` e `predict()`.

2. **Tupla `(point_forecast, samples)`**: Tutti i modelli restituiscono sia un point forecast (mediana o media) sia campioni probabilistici opzionali. I modelli deterministici (Seasonal Naive) restituiscono `samples=None`. Questo permette al runner di calcolare metriche probabilistiche (CRPS, WQL) solo quando disponibili.

3. **`fit()` restituisce `self`**: Abilita il method chaining (`model.fit(data).predict(...)`).

4. **`_is_fitted` flag**: Previene errori silenziosi. Senza questo check, un modello foundation non inizializzato produrrebbe un `NullPointerError` criptico; con il flag si ottiene un messaggio chiaro.

---

### 3.2 SeasonalNaiveModel

**File**: `statistical.py`

**Cosa fa**: Prevede che il futuro sia uguale allo stesso momento del ciclo stagionale precedente.

```python
forecast[i] = context[len(context) - seasonality + (i % seasonality)]
```

**Come lo fa**: Per ogni step futuro `i`, copia il valore dal context che si trova esattamente `seasonality` step prima. Se `i >= seasonality`, ricomincia il ciclo con il modulo.

**Perche' cosi'**: Il Seasonal Naive e' il **baseline di riferimento** per la metrica MASE. Per definizione, MASE = 1 per questo modello. Se nessun altro modello lo batte, non vale la pena usare modelli piu' complessi. Il default `seasonality=168` (una settimana di ore) cattura sia il ciclo giornaliero che quello settimanale (weekend vs. feriali).

`fit()` e' un no-op (imposta solo `_is_fitted=True`) perche' il modello non ha parametri da apprendere.

`predict()` restituisce `samples=None` perche' il modello e' puramente deterministico — non produce stime di incertezza.

---

### 3.3 ARIMAModel

**File**: `statistical.py`

**Cosa fa**: Applica un modello SARIMA (Seasonal ARIMA) al context per generare previsioni.

**Come lo fa**:
1. Tronca il context a `max_context` osservazioni (default: 2016 = 12 settimane).
2. Se `auto=True`, usa `pmdarima.auto_arima` per selezionare automaticamente l'ordine.
3. Altrimenti costruisce un modello con ordine fisso e lo fitta sul context.
4. Genera point forecast con `model.predict()`.
5. Genera intervalli di confidenza all'80% e approssima sample Gaussiani.

**Perche' cosi'**:

- **`max_context=2016`**: ARIMA ha complessita' O(n^2) o superiore nel fitting. Con 26k ore di training impiegherebbe minuti; con 2016 ore impiega secondi. Questo limite e' un compromesso accettabile perche' ARIMA usa comunque solo la storia recente.

- **ARIMA refittato ad ogni `predict()`**: A differenza dei TSFM che vengono caricati una volta, ARIMA e' refittato sul context di ogni finestra. Questo e' intenzionale: ARIMA adatta i suoi parametri alla distribuzione locale, che cambia nel tempo.

- **`fit()` e' un no-op**: Il fitting reale avviene in `predict()` perche' ARIMA si fitta sul context specifico di ogni finestra, non sull'intero train set.

- **Approssimazione campioni da CI**: ARIMA produce intervalli di confidenza, non sample. Per uniformare l'output con i TSFM, approssimiamo una Gaussiana con `std ≈ (upper - lower) / 2.56`, dove 2.56 e' la larghezza dell'intervallo 80% per una distribuzione normale standard (da -1.28σ a +1.28σ).

- **Lazy import di `pmdarima`**: La libreria e' opzionale. L'import dentro `predict()` permette al codice di funzionare (e ai test di passare) anche senza pmdarima installato.

---

### 3.4 ChronosBoltModel

**File**: `chronos_bolt.py`

**Cosa fa**: Wrapper per Chronos-Bolt, un TSFM basato su architettura T5 encoder-decoder con input patch-based, sviluppato da Amazon.

**Come lo fa**:

`fit()`:
1. Importa `ChronosBoltPipeline` da `chronos` (lazy import).
2. Costruisce l'ID del modello HuggingFace (`"amazon/chronos-bolt-{size}"`).
3. Carica i pesi pre-addestrati con `from_pretrained()`.

`predict()`:
1. Converte il context in tensore PyTorch con dimensione batch aggiunta (`unsqueeze(0)`).
2. Richiede le previsioni per i quantili [0.1, 0.5, 0.9].
3. Usa il quantile 0.5 (mediana) come point forecast.
4. Approssima sample Gaussiani dal range inter-quantile.

**Perche' cosi'**:

- **Lazy import**: `chronos-forecasting` e' una dipendenza pesante (~500 MB con le dipendenze). L'import dentro `fit()` permette di usare il resto del pacchetto senza averlo installato. I test usano mock al posto dell'import reale.

- **Quantili invece di sample**: Chronos-Bolt e' progettato per produrre direttamente quantili (piu' efficiente). Per uniformita' con gli altri modelli che producono sample, generiamo sample sintetici dalla distribuzione approssimata. Il valore `2.56` nel denominatore viene dal fatto che per una Gaussiana i quantili 0.1 e 0.9 distano circa 2.56 deviazioni standard.

- **`unsqueeze(0)` per batch dim**: La pipeline Chronos si aspetta un batch di serie. Per una singola serie aggiungiamo la dimensione batch manualmente.

- **Varianti di dimensione**: `tiny` (8M), `mini` (21M), `small` (48M), `base` (205M). Il default `"base"` offre il miglior rapporto accuratezza/velocita' su GPU T4.

---

### 3.5 Chronos2Model

**File**: `chronos2.py`

**Cosa fa**: Wrapper per Chronos-2, un TSFM encoder-only con group attention, successore di Chronos.

**Come lo fa**:

`fit()`: Carica `ChronosPipeline` con il modello `"amazon/chronos-t5-base"`.

`predict()`:
1. Converte il context in tensore.
2. Chiama `pipeline.predict()` che restituisce sample Monte Carlo.
3. Calcola la mediana dei sample come point forecast.

**Perche' diverso da Chronos-Bolt**: Chronos-2 usa la pipeline originale `ChronosPipeline` (non `ChronosBoltPipeline`) e produce sample invece di quantili. Questo e' piu' flessibile ma piu' lento. La mediana e' piu' robusta della media come point forecast perche' e' meno sensibile a sample estremi.

---

### 3.6 LagLlamaModel

**File**: `lag_llama.py`

**Cosa fa**: Wrapper per Lag-Llama, il primo TSFM open-source basato su architettura decoder-only (simile a LLaMA) con lag features.

**Come lo fa**:

`fit()`:
1. Aggiunge la directory del repo Lag-Llama a `sys.path` (il modello non e' un pacchetto pip).
2. Carica il checkpoint PyTorch e ne estrae gli iperparametri.
3. Costruisce un `LagLlamaEstimator` con i parametri del checkpoint.
4. Crea il modulo Lightning e la pipeline di trasformazione.

`predict()`:
1. Aggiorna `prediction_length` sull'estimator.
2. Costruisce un `PyTorchPredictor` di GluonTS.
3. Converte il context in un `PandasDataset`.
4. Genera previsioni con sampling.

**Perche' cosi'**:

- **`sys.path` manipulation**: Lag-Llama non e' un pacchetto installabile via pip — bisogna clonare il repo e importare direttamente. Aggiungere la directory al path e' la soluzione raccomandata dagli autori.

- **Parametri dal checkpoint**: Invece di hardcodare gli iperparametri (n_layer, n_head, ecc.), li leggiamo dal checkpoint stesso. Questo rende il codice compatibile con versioni future del modello.

- **`prediction_length` modificato a ogni predict**: L'estimator viene creato con un placeholder di 24, poi aggiornato. Questo perche' GluonTS richiede che il prediction_length sia noto alla costruzione del predictor, ma noi vogliamo testare orizzonti diversi.

- **`PandasDataset.from_long_dataframe`**: GluonTS usa un formato dati proprio. Questa conversione trasforma la nostra `pd.Series` nel formato richiesto.

---

### 3.7 ProphetModel

**File**: `prophet_model.py`

**Cosa fa**: Wrapper per Facebook Prophet, un modello additivo con componenti di trend, stagionalita' multipla e festivita'.

**Come lo fa**:

`fit()`:
1. Converte la serie nel formato Prophet (`DataFrame` con colonne `ds` e `y`).
2. Limita il training a `max_train_hours` osservazioni per velocita'.
3. Fitta il modello con stagionalita' giornaliera, settimanale e annuale.

`predict()`:
1. Crea un `future_df` con le date da prevedere.
2. Chiama `model.predict()` che restituisce `yhat`, `yhat_lower`, `yhat_upper`.
3. Approssima sample Gaussiani dagli intervalli di confidenza.

**Perche' cosi'**:

- **`max_train_hours=8760` (1 anno)**: Prophet scala linearmente ma con costante alta — su 26k ore impiega minuti. Un anno di dati e' sufficiente per catturare tutte e tre le stagionalita'.

- **`suppress_logging=True`**: Prophet usa Stan per l'inferenza e produce output verboso di default. Lo sopprimiamo per mantenere puliti i log del benchmark.

- **`changepoint_prior_scale=0.05`**: Controlla la flessibilita' del trend. Il default Prophet e' 0.05; valori piu' alti permettono al trend di cambiare piu' frequentemente. Per il carico elettrico, che ha trend stabile, il default e' appropriato.

- **Approssimazione campioni**: Identica alla logica ARIMA — Prophet produce CI, non sample. Usiamo la stessa formula Gaussiana.

---

## 4. Modulo `evaluation`

**File**: `src/energy_benchmark/evaluation/`

### 4.1 Metriche

**File**: `metrics.py`

#### `mae(y_true, y_pred) -> float`

**Cosa fa**: Mean Absolute Error, la media degli errori assoluti.

**Formula**: `MAE = mean(|y_true - y_pred|)`

**Perche'**: E' la metrica piu' intuitiva — esprime l'errore medio in MW. Facile da comunicare a stakeholder non tecnici.

---

#### `rmse(y_true, y_pred) -> float`

**Cosa fa**: Root Mean Squared Error.

**Formula**: `RMSE = sqrt(mean((y_true - y_pred)^2))`

**Perche'**: Penalizza gli errori grandi piu' del MAE. Utile in contesti energetici dove un singolo errore grande (es. sottostima di 5 GW) ha conseguenze molto peggiori di tanti piccoli errori.

---

#### `mase(y_true, y_pred, y_train, seasonality=24) -> float`

**Cosa fa**: Mean Absolute Scaled Error — normalizza l'errore rispetto all'errore di un naive stagionale.

**Formula**:

```
scaling_factor = mean(|y_train[t] - y_train[t - seasonality]|)
MASE = mean(|y_true - y_pred|) / scaling_factor
```

**Interpretazione**: MASE < 1 significa che il modello batte il naive stagionale. MASE = 0.8 significa "20% migliore del naive".

**Perche' cosi'**:

- **Scale-free**: A differenza di MAE/RMSE, MASE non dipende dalla scala dei dati. Un MASE di 0.85 ha lo stesso significato su ERCOT (decine di GW) e su una rete municipale (decine di MW).

- **`seasonality=24`** (default): Usa il naive giornaliero come riferimento. La scelta di 24 (e non 168) e' standard nella letteratura per serie orarie.

- **Protezione da divisione per zero**: Se `scaling_factor < 1e-9` (dati costanti), restituisce `inf`. Questo evita falsi positivi su dati triviali.

- **Calcolato su `y_train`**: Il scaling factor usa il training set, non il test set. Questo e' cruciale: se lo calcolassimo sul test set, il denominatore cambierebbe a ogni finestra, rendendo i risultati non confrontabili.

---

#### `crps(y_true, forecast_samples) -> float`

**Cosa fa**: Continuous Ranked Probability Score — misura la qualita' della distribuzione predetta, non solo del punto.

**Formula**: `CRPS = E|X - y| - 0.5 * E|X - X'|` dove X, X' sono campioni dalla distribuzione predetta.

**Come lo fa**:
1. Prova a usare `properscoring.crps_ensemble()` (implementazione C ottimizzata).
2. Se non disponibile, fallback a implementazione Python manuale.

**Perche' cosi'**: Il CRPS e' lo standard per valutare previsioni probabilistiche. A differenza del MAE, penalizza sia la bias che la varianza della distribuzione. Il fallback manuale ha complessita' O(n * num_samples^2) — lento, ma funziona senza dipendenze extra.

---

#### `weighted_quantile_loss(y_true, y_pred_quantiles, quantiles) -> float`

**Cosa fa**: Media pesata della pinball loss per ciascun quantile.

**Formula**: Per ciascun quantile q:

```
loss_q = mean(max(q * e, (q-1) * e))  dove e = y_true - y_pred_q
```

**Come lo fa**: Per ogni quantile, calcola la pinball loss (asimmetrica: penalizza di piu' gli errori nella direzione sbagliata). Media su tutti i quantili.

**Perche' cosi'**: WQL e' la metrica standard per forecast quantilici, usata in competizioni come M5. I quantili default [0.1, 0.5, 0.9] corrispondono al lower bound, mediana e upper bound — le previsioni piu' rilevanti per il mercato energetico (worst case, best estimate, best case).

---

### 4.2 BenchmarkRunner

**File**: `benchmark.py`

#### Strutture dati

```python
@dataclass
class ForecastResult:
    model_name, horizon, context_length, window_idx,
    point_forecast, actual, samples, elapsed_seconds
```

**Cosa fa**: Contenitore immutabile per il risultato di una singola finestra di forecast.

**Perche' un dataclass**: Semplice, serializzabile, auto-genera `__init__` e `__repr__`. Manteniamo sia `point_forecast` che `actual` per poter fare analisi post-hoc (es. errore per ora del giorno).

```python
@dataclass
class BenchmarkResults:
    records: List[Dict]      # risultati aggregati (1 riga per model/horizon/ctx)
    forecasts: List[ForecastResult]  # risultati per finestra (per analisi fine)
```

**Perche' due strutture**: `records` e' ottimizzato per creare un DataFrame riassuntivo. `forecasts` mantiene i dettagli per visualizzazioni e debug.

---

#### `class BenchmarkRunner`

**Attributo `METRIC_FNS`**:

```python
METRIC_FNS = {
    "mae": lambda yt, yp, **kw: M.mae(yt, yp),
    "mase": lambda yt, yp, **kw: M.mase(yt, yp, kw["y_train"], seasonality=24),
    ...
}
```

**Cosa fa**: Mappa nomi di metriche (stringhe) a funzioni callable.

**Perche' cosi'**: Permette di configurare quali metriche calcolare via YAML (`metrics: [mae, rmse, mase]`) senza codice condizionale. Le lambda con `**kw` accettano argomenti extra (come `y_train`, `samples`) che non tutte le metriche usano.

**Per CRPS e WQL**: Se `samples` e' `None` (es. SeasonalNaive), restituiscono `NaN`. Questo e' preferibile a sollevare un'eccezione perche' permette al benchmark di continuare con le altre metriche.

---

#### `BenchmarkRunner.run(train, test, rolling_config) -> BenchmarkResults`

**Cosa fa**: Esegue il benchmark completo: per ogni modello, per ogni context length, per ogni orizzonte, esegue la rolling window evaluation.

**Come lo fa**:
1. Per ogni modello, verifica che sia fittato; se no, lo fitta.
2. Per ogni combinazione (context_length, horizon), chiama `_evaluate_rolling()`.
3. Aggrega i risultati per finestra in un record medio.
4. Logga i risultati incrementalmente.

**Perche' cosi'**:

- **Triple loop (model x ctx x horizon)**: Massimizza il riuso — il modello viene caricato una volta e testato su tutte le combinazioni.

- **Auto-fitting**: Se un modello non e' ancora fittato, il runner lo fitta automaticamente. Questo semplifica l'uso in notebook dove potresti dimenticare di chiamare `fit()`.

---

#### `BenchmarkRunner._evaluate_rolling(...) -> List[Dict[str, float]]`

**Cosa fa**: Esegue la rolling window evaluation per una singola combinazione (modello, orizzonte, context).

**Come lo fa**:
1. Concatena train + test in una serie completa.
2. Per ogni finestra `w` (da 0 a `max_windows`):
   - Calcola l'inizio della previsione: `test_start_idx + w * step_size`.
   - Estrae il context (i `context_length` step precedenti).
   - Estrae i valori reali per il confronto.
   - Chiama `model.predict()` con timing.
   - Calcola tutte le metriche richieste.
3. Se una finestra fallisce (eccezione), la salta con un warning.

**Perche' cosi'**:

- **Rolling window**: E' lo standard nella letteratura di forecasting. Valutare su una sola finestra darebbe risultati non robusti — condizioni meteo anomale in quella settimana distorcerebbero i risultati. 30 finestre danno una stima statistica affidabile.

- **`step_size=24`**: Avanza di 24 ore (1 giorno) tra una finestra e la successiva. Questo bilancia granularita' (piu' finestre = stima piu' robusta) e velocita' (meno finestre = benchmark piu' veloce).

- **Concatenazione train+test**: Il context di una finestra puo' estendersi nel train set (le prime finestre di test hanno bisogno di storia precedente). Concatenare evita problemi di boundary.

- **`max(0, forecast_start - context_length)`**: Per le primissime finestre, il context potrebbe richiedere dati precedenti al train set. Il `max(0, ...)` previene indici negativi.

- **Eccezioni gestite per finestra**: Se un modello fallisce su una finestra (es. divergenza numerica di ARIMA), il benchmark continua con le altre finestre. Questo e' preferibile a bloccare tutto il benchmark per un singolo errore.

- **`time.perf_counter()`**: Misura il tempo di inferenza ad alta risoluzione (non il wall-clock time). Utile per il confronto di velocita' tra modelli.

---

## 5. Modulo `visualization`

**File**: `src/energy_benchmark/visualization/plots.py`

#### `plot_comparison(results_df, metric, save_path, figsize) -> Figure`

**Cosa fa**: Grafico a barre raggruppate che confronta i modelli per ogni orizzonte.

**Come lo fa**: Usa `pivot_table` per riorganizzare il DataFrame (righe=modelli, colonne=orizzonti) e poi `DataFrame.plot(kind="bar")`.

**Perche' cosi'**: Il bar chart raggruppato e' il formato piu' immediato per confrontare performance relative. Ogni cluster di barre mostra un modello, i colori distinguono gli orizzonti. Questo permette di rispondere a colpo d'occhio a "quale modello e' migliore?" e "come cambia la performance al crescere dell'orizzonte?".

---

#### `plot_forecasts(actual, predictions, start_idx, length, ...) -> Figure`

**Cosa fa**: Sovrappone le previsioni di piu' modelli alla serie reale.

**Come lo fa**: Plotta la serie reale in nero (spesso), poi ogni modello con colore e trasparenza diversi.

**Perche' cosi'**: La visualizzazione diretta della previsione vs realta' e' essenziale per il debugging qualitativo — le metriche aggregate possono nascondere pattern sistematici (es. un modello che sottostima sempre i picchi).

---

#### `plot_metric_heatmap(results_df, metric, ...) -> Figure`

**Cosa fa**: Heatmap con seaborn che mostra il valore numerico della metrica per ogni combinazione (modello, orizzonte).

**Come lo fa**: Pivot table → `sns.heatmap()` con annotazioni numeriche e colormap `YlOrRd` (giallo=bene, rosso=male).

**Perche' cosi'**: La heatmap permette di individuare immediatamente dove un modello eccelle o fatica. Il formato numerico annotato e' adatto per inclusione diretta in paper o presentazioni.

---

#### `plot_probabilistic_forecast(actual, point_forecast, samples, quantiles, ...) -> Figure`

**Cosa fa**: Visualizza una singola previsione con bande di incertezza.

**Come lo fa**:
1. Plotta la serie reale in nero.
2. Plotta il point forecast in blu.
3. Se ci sono sample, calcola i quantili richiesti e usa `fill_between()` per la banda colorata.

**Perche' cosi'**: Le bande di incertezza sono cruciali per la valutazione dei forecast probabilistici. Una banda stretta e' utile solo se e' ben calibrata (contiene ~80% dei valori reali). Una banda troppo larga e' inutile. Questa visualizzazione permette di valutarlo visivamente.

---

## 6. Scripts

### 6.1 download_data.py

**File**: `scripts/download_data.py`

**Cosa fa**: CLI per scaricare e cachare i dati ERCOT senza eseguire il benchmark.

**Argomenti**:
- `--years`: Anni da scaricare (default: 2020-2024).
- `--data-dir`: Directory di cache (default: `data/raw`).
- `--force`: Ri-scarica anche se i file esistono.

**Perche' come script separato**: Separa il download (lento, richiede internet) dal benchmark (veloce, offline). In Colab, l'utente scarica una volta e poi riesegue il benchmark piu' volte senza ri-scaricare.

---

### 6.2 run_benchmark.py

**File**: `scripts/run_benchmark.py`

**Cosa fa**: Esegue il benchmark completo da linea di comando.

**Come lo fa**:
1. Carica la configurazione YAML.
2. Istanzia `ERCOTLoader` e carica/processa i dati.
3. Chiama `_build_models()` per creare i modelli abilitati nel config.
4. Istanzia `BenchmarkRunner` e lancia `run()`.
5. Salva i risultati in CSV e genera figure PNG.

#### `_build_models(model_cfg, force_cpu) -> list`

**Cosa fa**: Crea la lista di modelli in base alla configurazione YAML.

**Come lo fa**: Per ogni modello nel config, verifica se `enabled: true` e lo istanzia con i parametri specificati.

**Perche' cosi'**: Disaccoppia la configurazione dall'istanziazione. Cambiare i modelli da testare richiede solo modificare il YAML, non il codice Python. Il flag `--cpu` forza tutti i modelli su CPU — utile per sviluppo locale senza GPU.

---

## 7. Demo Streamlit

**File**: `demo/streamlit_app.py`

**Cosa fa**: Applicazione web interattiva per esplorare i dati e lanciare il benchmark.

**Come lo fa**: Tre tab:

1. **Data Explorer**: Carica e visualizza la serie ERCOT con split colorati e profilo giornaliero.
2. **Run Benchmark**: Permette di selezionare modelli, orizzonte e numero di finestre dalla sidebar, poi esegue il benchmark con un click.
3. **Results**: Mostra tabella, grafici a barre e un esempio di forecast.

**Scelte implementative**:

- **`@st.cache_data`**: I dati ERCOT vengono scaricati e processati una sola volta, poi cachati nella sessione Streamlit. Questo evita ri-download ad ogni interazione.

- **`st.session_state["results"]`**: I risultati del benchmark vengono salvati nello stato di sessione per persistere tra i tab.

- **Modelli in `try/except`**: Se `chronos-forecasting` non e' installato, il modello viene silenziosamente saltato con un warning nell'UI. L'utente puo' sempre usare il Seasonal Naive senza dipendenze extra.

- **`plt.close()` dopo `st.pyplot()`**: Necessario in Streamlit per evitare che le figure si accumulino in memoria.

---

## 8. Configurazione YAML

**File**: `configs/benchmark_config.yaml`

**Struttura**:

```yaml
data:           # Sorgente dati, anni, colonna target, split dates
models:         # Quali modelli abilitare e con quali parametri
benchmark:      # Orizzonti, context lengths, rolling window config
metrics:        # Lista di metriche da calcolare
output:         # Directory per risultati e figure
```

**Perche' YAML**: Formato leggibile, standard per configurazione ML. Permette di versionare esperimenti diversi (es. `config_fast.yaml` con meno finestre per testing, `config_full.yaml` per il benchmark finale).

**`prophet.enabled: false`**: Prophet e' disabilitato di default perche' richiede una dipendenza pesante (`pystan`/`cmdstanpy`) e non e' un modello fondazionale. Lo includiamo come opzione per completezza.

---

## 9. Testing

**File**: `tests/`

35 test totali distribuiti in 4 file:

### `test_data_loader.py` (10 test)

- **TestERCOTLoader**: Verifica la validazione degli anni, i default, e il parsing dei timestamp ERCOT (inclusi i casi speciali `24:00` e `DST`).
- **TestPreprocessing**: Verifica che il preprocessing elimini tutti i NaN, che la normalizzazione sia invertibile (roundtrip test), e che gli split non perdano dati.

### `test_metrics.py` (8 test)

- **TestPointMetrics**: Casi noti per MAE e RMSE (previsione perfetta = 0, errore noto).
- **TestMASE**: Verifica che il MASE sia ~1 per un naive e 0 per una previsione perfetta.
- **TestWQL**: Verifica che WQL = 0 per quantili perfetti e > 0 per errori.

### `test_models.py` (11 test)

- **TestBaseContract**: Verifica che `ForecastModel` non sia istanziabile direttamente (e' una ABC).
- **TestSeasonalNaive**: Verifica lunghezza output e che la previsione corrisponda al lag atteso.
- **TestARIMA, TestChronosBolt, TestChronos2, TestProphet**: Usano mock per le dipendenze esterne (`pmdarima`, `chronos`, `prophet`). I mock vengono iniettati via `patch.dict(sys.modules)` perche' gli import sono lazy (dentro `fit()`/`predict()`).

### `test_benchmark.py` (6 test)

- **TestBenchmarkRunner**: Verifica end-to-end su dati sintetici con SeasonalNaive. Controlla che il DataFrame di output abbia le colonne e righe corrette.
- **TestVisualization**: Smoke test — verifica solo che le funzioni di plotting non crashino (non verificano il contenuto visivo).

**Strategia di mocking**: I modelli fondazionali richiedono download di GB di pesi. Per i test, iniettiamo moduli finti con `patch.dict(sys.modules, {"chronos": mock_chronos})`. Questo simula l'intero modulo `chronos` con un `ModuleType` che espone solo le classi necessarie. I mock restituiscono tensori PyTorch con forma corretta, permettendo di testare tutta la logica di conversione e output senza i pesi reali.

---

## 10. Scelte Progettuali Trasversali

### Lazy imports per dipendenze opzionali

Tutte le dipendenze pesanti (`chronos`, `pmdarima`, `prophet`, `gluonts`) sono importate dentro i metodi, non a livello di modulo. Questo garantisce che:
- `pip install -e .` funzioni senza installare tutto
- I test passino senza GPU ne' modelli scaricati
- L'utente installi solo cio' che gli serve

### Logging al posto di print

Tutto il codice usa `logging.getLogger(__name__)` invece di `print()`. Questo permette di:
- Controllare la verbosita' dal chiamante (`logging.basicConfig(level=...)`)
- Filtrare per modulo
- Redirigere a file in produzione

### Parquet come formato di cache

I dati ERCOT scaricati vengono salvati in Parquet (non CSV/Excel) perche':
- 10x piu' veloce in lettura
- Preserva i tipi (DatetimeIndex, float64) senza ambiguita'
- Compressione trasparente

### Type hints ovunque

Ogni funzione pubblica ha type hints completi. Questo:
- Documenta l'API senza dover leggere l'implementazione
- Abilita il controllo statico con `mypy`
- Migliora l'autocompletamento nell'IDE

### Approssimazione Gaussiana dei campioni

Tre modelli (Chronos-Bolt, ARIMA, Prophet) producono intervalli/quantili ma non campioni nativi. Per uniformita', generiamo campioni approssimati:

```
std ≈ (upper - lower) / 2.56
samples ~ Normal(point, std)
```

Il valore 2.56 viene dalla distribuzione normale standard: i quantili 0.1 e 0.9 corrispondono a z = -1.28 e z = +1.28, quindi la distanza e' 2.56 sigma. Questa approssimazione e' ragionevole per serie di carico elettrico che hanno distribuzioni approssimativamente simmetriche.

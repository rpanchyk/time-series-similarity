# Time Series Similarity

Дослідницька робота з пошуку подібності часових рядів (Forex OHLC).

## Вхідні дані

### Тікові дані (ticks)
Файл `data/ticks/<SYMBOL>_ticks.csv` містить тікові дані символу на ринку Форекс за певний період.

### Робочі дані (ohlc)
Набір файлів у папці `data/ohlc/`:
- `<SYMBOL>_<TIMEFRAME>.csv` — ціни Open, High, Low, Close.

Приклад для 4-годинного часового ряду:

```
2003-01-01 00:00:00,1.23456,1.23457,1.23458,1.23459
2003-01-01 04:00:00,1.23457,1.23458,1.23459,1.23460
```

Примітка: ціна має ту саму кількість знаків після коми, що й у тікових даних.

## Параметри

| Параметр | Сенс | Default |
|----------|------|---------|
| `window` | довжина query/match вікна (бари) | 20 |
| `stride` | крок ковзного вікна | 1 |
| `horizon` | бари **після** матчу для outcome | 20 |
| `top-k` | кількість найближчих аналогів | 10 |
| `price` | канал для уніваріантних метрик | close |

## Типи пошуку подібностей (`--search-type`)

Реєстр у `src/config.py` → `SEARCH_TYPES`. CLI: одне значення або список через кому (`dtw,pearson`).

| `--search-type` | Опис |
|-----------------|------|
| `z-euclidean` | z-нормалізована евклідова відстань по `--price` |
| `z-manhattan` | z-нормалізована L1 (Manhattan) по `--price` |
| `pearson` | відстань Пірсона: \(1 - \mathrm{corr}\) |
| `spearman` | рангова відстань Спірмена: \(1 - \mathrm{rank\ corr}\) |
| `cosine` | косинусна відстань: \(1 - \mathrm{cosine\ similarity}\) |
| `cid` | Complexity-Invariant Distance (z-нормалізована) |
| `dtw` | Dynamic Time Warping (z-нормалізована) |
| `softdtw` | Soft-DTW (z-нормалізована) |
| `logret` | z-евклід по log-returns від `--price` |
| `ohlc-joint` | евклід по `concat(z(close), z(high-low))` |
| `candle-shape` | евклід по ознаках тіла/тіней свічки |
| `dtw-multiv` | багатовимірний DTW по z-нормалізованих каналах OHLC |

Структурні метрики (`ohlc-joint`, `candle-shape`, `dtw-multiv`) ігнорують `--price`.  
Важкі типи: `dtw`, `softdtw`, `dtw-multiv` (час ~window²). У бектесті за замовчуванням виключені `softdtw` і `dtw-multiv` (`--exclude`).

## Задача

1. Підготувати OHLC з тіків.
2. Знайти подібні історичні вікна до query.
3. Оцінити, що було далі на `horizon` (таблиця + графік).
4. (Дослідження) Порівняти, який `--search-type` + `--predict` краще передбачає bias query.

## Реалізація

- Мова: Python
- CLI: `main.py`; логіка в `src/`
- Дослідницькі скрипти: `scripts/`

### Підготовка робочих даних
Виконується в багатопроцесорному режимі (`--workers`).

## Оцінка напряму (bias) після матчу

Після кожного збігу на горизонті `horizon` рахуються:
- `max_dev_high` — максимальне відхилення **вгору** від close кінця матчу, у **%**;
- `max_dev_low` — максимальне відхилення **вниз** (зазвичай від’ємне), у **%**;
- `max_dev_*_time` — час досягнення екстремуму (для first-touch).

Для одного результату (рядка top-k) **amplitude bias**:

```
up   = abs(max_dev_high)
down = abs(max_dev_low)
BIAS_MARGIN = 0.20   # відсоткові пункти

if up - down > BIAS_MARGIN:
    bias = LONG
elif down - up > BIAS_MARGIN:
    bias = SHORT
else:
    bias = SKIP
```

Еквівалентно: перевага однієї сторони > **0.20** в.п.; інакше SKIP.

**First-touch** (окрема ціль): хто настав раніше за часом — high чи low (`long` / `short` / `skip` при нічиї).

Примітки:
- Поля вже в процентах (`0.27` = 0.27%); додаткове `×100` не потрібне.
- Це **3-класова** задача (LONG/SHORT/SKIP): рівномірний рандом ≈ 33%, majority baseline ≈ 35–36% на EURUSD 1h. Монетка 50% — лише для 2 класів (first-touch).

## Бектест точності типів

Скрипт: `scripts/backtest_search_types.py`

Сітка за замовчуванням: випадкові `--start-date` (`--n-dates`, `--seed`), windows, `--skip-weekends`, `top-k=10`, `horizon=20`.  
Одні й ті самі дати застосовуються до **всіх** window / search-type у прогоні.

Метрики:
- **hit** — `predict == actual` (для query на його horizon);
- **soft** — частка top-k з bias == actual;
- для `signed-*` також **signed_agree**.

```bash
# Усі швидкі типи × усі predict, n=100
python scripts/backtest_search_types.py \
  --n-dates 100 --seed 42 --windows 10,20,30,40 \
  --predict all --exclude softdtw,dtw-multiv,dtw --tag full100

# Звужений план (лише обрані пари window × type × predict)
python scripts/backtest_search_types.py \
  --n-dates 400 --seed 42 \
  --window-plan scripts/amplitude_narrow_plan.json --tag narrow400
```

`--window-plan` — JSON:

```json
{
  "windows": {
    "10": [{"search_type": "ohlc-joint", "predict": ["gaussian-mean", "trimmed-mean"]}],
    "40": [
      {"search_type": "pearson", "predict": ["rank-mean-weighted"]},
      {"search_type": "z-euclidean", "predict": ["rank-mean-weighted"]}
    ]
  }
}
```

Інші прапорці: `--margin-conf`, `--workers`, `--tag` (суфікс імен у `output/`).

Огляд одного query по всіх типах (не walk-forward): `scripts/analyze_direction.py`  
(увага: там старіше правило dominance 1.25× / min-move 0.05%; канонічне для бектесту — `BIAS_MARGIN` вище).

### `--predict` — як top-k стає прогнозом

`--predict all` рахує **всі** режими на тих самих top-k (дешево). Зараз **22** режими.

**Голосування міток bias (amplitude):**

| Режим | Прогноз | Ранжування |
|-------|---------|------------|
| `mode` | найчастіший bias | hit, soft |
| `soft` | label = mode; окремо **soft%** = частка top-k == actual | soft |
| `margin` | mode лише якщо \((n_1-n_2)/k \ge\) `--margin-conf` | hit |
| `distance-weighted` | mode з вагами \(1/(d-d_{\min})\) | hit |
| `softmax` | mode з softmax(\(-d\)) | hit |
| `softmax-margin` | softmax-ймовірності; SKIP якщо \(P_1-P_2 <\) margin | hit |
| `rank-weighted` | mode з вагами за рангом (ближчий важчий) | hit |
| `gaussian` | mode з Gaussian-ядром по distance | hit |
| `recency-weighted` | mode з \(\exp(-\mathrm{age}/365\mathrm{d})\) | hit |

`soft` ≠ `softmax`: `soft` — метрика частки + pred=mode; `softmax` — зважене голосування по distance.

**Агрегати величин → bias rule:**

| Режим | Прогноз | Ранжування |
|-------|---------|------------|
| `mean` | bias(mean\|high\|, mean\|low\|) | hit |
| `median` | bias(median\|high\|, median\|low\|) | hit |
| `trimmed-mean` | bias(10% trimmed mean) | hit |
| `mean-weighted` | bias(distance-weighted means) | hit |
| `rank-mean-weighted` | bias(rank-weighted means) | hit |
| `gaussian-mean` | bias(Gaussian-weighted means) | hit |
| `recency-mean-weighted` | bias(recency-weighted means) | hit |
| `quantile-75` | bias(P75\|high\|, P75\|low\|) | hit |
| `signed-score` | \(s=\mathrm{mean}(\|h\|-\|l\|)\) → bias(s) | signed_agree, hit |
| `signed-median` | \(s=\mathrm{median}(\|h\|-\|l\|)\) → bias(s) | signed_agree, hit |

**First-touch** (actual = хто раніше: high_time чи low_time):

| Режим | Прогноз |
|-------|---------|
| `first-touch` | mode першого екстремуму по top-k |
| `first-touch-weighted` | distance-weighted mode first-touch |
| `first-touch-margin` | first-touch mode з margin abstain |

Примітка: `mean` і `signed-score` дають ту саму жорстку мітку (\(s=\mathrm{up}-\mathrm{down}\)); відрізняються ранжуванням.

## Висновки з бектестів (EURUSD 1h, seed=42)

Умови: `horizon=20`, `top-k=10`, `--skip-weekends`; без важких DTW (де зазначено).

| Прогін | Що перевіряли | Amplitude overall |
|--------|---------------|-------------------|
| n=20, all predicts | сітка window×type | кращі пари ~40–65% на малих комірках — **overfit** |
| n=100, full grid | 9 типів × 22 predict, w=10/20/30/40 | best overall ~**41.5%** (`rank-mean-weighted`+pearson); majority ~36% |
| n=200 focused | w10 ohlc-joint mean-family; w40 pearson/z-euc + rank-mean-weighted | **~40%**; дати 101–200 сильно гірші |
| n=400 narrow | ті самі вузькі пари | **~35%** ≈ majority baseline |

Спостереження:
1. Лідери на **малому n** майже не відтворюються на **великому n** (класичний selection bias / overfit по сітці).
2. Mean-family агрегати (`rank-mean-weighted`, `gaussian-mean`, `trimmed-mean`) трохи кращі за простий `mode`, але lift малий.
3. Window **20** на цій цілі слабкий; умовні «кращі» зони були w10 / w40 — і вони теж зійшли до baseline при n=400.
4. First-touch (2 класи) дає вищий абсолютний hit (~50–55%), але це інша ціль і ближче до монетки.

Практичний статус: **стабільного предиктивного edge для amplitude bias на EURUSD 1h цією подібністю не підтверджено** на n=400. CLI пошуку аналогів лишається корисним для дослідження / візуалізації, не як готовий сигнал напряму.

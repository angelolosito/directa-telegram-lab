# Directa Telegram Trading Lab

Bot sperimentale per generare segnali **paper trading** su strumenti quotati su Borsa Italiana / ETFplus / Euronext Milan, con notifiche Telegram.

> **Nota importante**: questo progetto non invia ordini reali e non si collega al conto Directa. Serve per testare regole operative in modo disciplinato. Le decisioni reali restano manuali.

## Cosa fa

- Scarica dati giornalieri tramite `yfinance`.
- Applica timeout e retry brevi al download dati, così un ticker lento non blocca l'intero run.
- Calcola indicatori tecnici: SMA 20/50/200, RSI 14, ATR 14, volume medio.
- Genera segnali con due strategie:
  - Trend + Pullback
  - Breakout controllato
- Assegna uno score 0-100 ai segnali e mostra la classifica dei migliori candidati.
- Valuta il regime di mercato con benchmark globali: se il mercato è fragile blocca nuovi ingressi o alza la soglia score.
- Rilegge ogni segnale con una checklist opportunità: mercato, trend, timing, momentum, volumi, rischio e costi.
- Mostra anche setup quasi pronti, così puoi vedere cosa sta maturando prima del trigger operativo.
- Tiene un diario dei segnali e valuta dopo 5/10/20/40 sedute se il setup era davvero valido.
- Può usare il diario come feedback prudenziale: setup storicamente deboli vengono penalizzati nello score.
- Confronta ogni candidato con un benchmark di riferimento e premia solo gli strumenti con buona forza relativa.
- Usa un selettore finale di portafoglio per evitare segnali troppo simili per settore, area o ruolo.
- Produce un report di calibrazione per capire se le soglie sono troppo rigide, troppo permissive o concentrate.
- Simula acquisti/vendite con paper trading su SQLite.
- Applica vincoli di rischio:
  - Capitale laboratorio: 1.000 €
  - Rischio massimo per trade: 25 €
  - Perdita massima mensile: 100 €
  - Massimo 2 posizioni aperte
  - Massimo 6 ingressi al mese
  - Cooldown di 5 giorni dopo uno stop loss sullo stesso strumento
- Stima commissioni Directa con modello configurabile.
- Invia alert Telegram.
- Salva report giornaliero in `reports/`.

## Struttura

```text
directa-telegram-lab/
├── .github/workflows/daily-alerts.yml
├── config.yaml
├── watchlist.yaml
├── main.py
├── requirements.txt
├── src/
│   ├── backtest.py
│   ├── allocation.py
│   ├── calibration.py
│   ├── config.py
│   ├── costs.py
│   ├── currency.py
│   ├── data_provider.py
│   ├── indicators.py
│   ├── market_regime.py
│   ├── opportunity.py
│   ├── paper_portfolio.py
│   ├── relative_strength.py
│   ├── report.py
│   ├── signal_journal.py
│   ├── strategy.py
│   └── telegram_notifier.py
├── state/
│   └── trading_lab.sqlite
└── reports/
```

## Setup locale

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --dry-run
```

`--dry-run` usa una copia temporanea del database: mostra chiusure, trailing stop e nuovi segnali simulati, ma non salva modifiche in `state/`, `data/` o `reports/`.

Per eseguire un backtest storico sulle regole attive:

```bash
python main.py --backtest
```

Il report viene stampato in console e, se `save_reports` è attivo, salvato in `reports/backtest_YYYY-MM-DD.md`.

Per eseguire una calibrazione sulle regole attive:

```bash
python main.py --calibration-report
```

Il report viene stampato in console e salvato in `reports/calibration_YYYY-MM-DD.md`. Serve a capire frequenza operativa, bucket migliori/deboli, settori, aree geografiche e strumenti della watchlist rimasti silenziosi.

Per eseguire i test automatici:

```bash
python -m unittest discover -s tests
```

Per leggere il report del diario intelligente:

```bash
python main.py --learning-report
```

Per inviare Telegram in locale:

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC..."
export TELEGRAM_CHAT_ID="123456789"
python main.py
```

## Setup GitHub Actions

1. Crea un repository GitHub privato.
2. Carica tutti i file del progetto.
3. Vai su **Settings → Secrets and variables → Actions → New repository secret**.
4. Aggiungi:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Vai su **Actions** e avvia manualmente il workflow `Daily Trading Alerts`.
6. Controlla che arrivi il messaggio Telegram.

Il workflow gira da lunedì a venerdì alle 18:15 Europe/Rome.

## Come creare il bot Telegram

1. Apri Telegram.
2. Cerca `@BotFather`.
3. Scrivi `/newbot`.
4. Dai un nome e uno username al bot.
5. Copia il token e salvalo come secret GitHub `TELEGRAM_BOT_TOKEN`.
6. Scrivi almeno un messaggio al tuo bot.
7. Per recuperare il tuo chat id puoi usare:

```text
https://api.telegram.org/bot<TOKEN>/getUpdates
```

Cerca il valore `chat.id` e salvalo come `TELEGRAM_CHAT_ID`.

## Modifica watchlist

Apri `watchlist.yaml` e aggiungi/rimuovi strumenti. I ticker sono in formato Yahoo Finance, ad esempio:

```yaml
- symbol: ENEL.MI
  name: Enel
  type: stock
```

## Modifica rischio

Apri `config.yaml`:

```yaml
data:
  request_timeout_seconds: 6
  process_timeout_seconds: 20
  download_retries: 0

backtest:
  lookback_days: 900
  min_rows_required: 220
  max_new_positions_per_day: 1

learning:
  enabled: true
  horizons_sessions: [5, 10, 20, 40]
  primary_horizon_sessions: 20
  min_bucket_count: 2
  adaptive_feedback_enabled: true
  adaptive_min_samples: 5
  adaptive_penalty_points: 6
  adaptive_bonus_points: 3

relative_strength:
  enabled: true
  lookback_sessions: 60
  default_benchmark: SWDA.MI
  benchmark_by_type:
    stock: EXW1.MI
    etf: SWDA.MI
  weak_threshold_pct: -2.0
  strong_threshold_pct: 2.0

market_regime:
  enabled: true
  neutral_score_boost: 5
  risk_off_score_boost: 15
  block_new_positions_when_risk_off: true
  benchmarks:
    - symbol: SWDA.MI
      name: iShares Core MSCI World UCITS ETF
    - symbol: CSSPX.MI
      name: iShares Core S&P 500 UCITS ETF

opportunity:
  enabled: true
  min_decision_score: 62
  ideal_pullback_distance_pct: 2.5
  max_pullback_distance_pct: 4.5
  ideal_breakout_extension_atr: 0.8
  max_breakout_extension_atr: 1.4
  max_cost_pct: 5.0

risk:
  initial_capital: 1000
  risk_per_trade: 25
  monthly_loss_limit: 100
  max_open_positions: 2
  max_trades_per_month: 6
  cooldown_after_stop_days: 5

strategy:
  min_signal_score: 60
  near_breakout_pct: 1.5
  setup_watch_min_score: 50
```

`min_signal_score` blocca i segnali tecnicamente validi ma qualitativamente deboli. Lo score considera forza del trend, RSI, rischio percentuale, volumi, rapporto rischio/rendimento e incidenza dei costi.
`market_regime` controlla il contesto generale: in mercato neutrale alza la soglia score, in mercato fragile può bloccare nuovi ingressi paper.
`opportunity` evita di inseguire prezzi troppo estesi: un segnale può diventare WATCH se il timing non è pulito, anche quando la strategia tecnica lo aveva generato.
`setup_watch_min_score` e `near_breakout_pct` alimentano il radar dei setup quasi pronti nella classifica candidati.
`learning` alimenta il diario intelligente in `data/signal_journal.csv` e `data/signal_evaluations.csv`: col tempo il bot misura quali setup hanno funzionato meglio. Il feedback adattivo si attiva solo dopo un numero minimo di casi simili.
`relative_strength` confronta titolo/ETF con un benchmark: se uno strumento resta debole rispetto al mercato, il suo score viene ridotto anche se il setup tecnico sembra valido.
Il cooldown post-stop evita di rientrare subito su un titolo appena chiuso male, mentre `max_trades_per_month` limita l'overtrading del laboratorio.
`process_timeout_seconds` è il taglio duro per singolo ticker: se Yahoo/YFinance resta appeso, quel simbolo viene saltato e il run continua.

## Disclaimer

Il codice è didattico e sperimentale. Non costituisce consulenza finanziaria, raccomandazione personalizzata, sollecitazione all'investimento o garanzia di rendimento. Verifica sempre dati, costi, fiscalità, liquidità e idoneità dello strumento prima di operare con denaro reale.

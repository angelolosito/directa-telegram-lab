# Directa Telegram Trading Lab

Bot sperimentale per generare segnali **paper trading** su strumenti quotati su Borsa Italiana / ETFplus / Euronext Milan, con notifiche Telegram.

> **Nota importante**: questo progetto non invia ordini reali e non si collega al conto Directa. Serve per testare regole operative in modo disciplinato. Le decisioni reali restano manuali.

## Cosa fa

- Scarica dati giornalieri tramite `yfinance`.
- Calcola indicatori tecnici: SMA 20/50/200, RSI 14, ATR 14, volume medio.
- Genera segnali con due strategie:
  - Trend + Pullback
  - Breakout controllato
- Simula acquisti/vendite con paper trading su SQLite.
- Applica vincoli di rischio:
  - Capitale laboratorio: 1.000 €
  - Rischio massimo per trade: 25 €
  - Perdita massima mensile: 100 €
  - Massimo 2 posizioni aperte
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
│   ├── config.py
│   ├── costs.py
│   ├── data_provider.py
│   ├── indicators.py
│   ├── paper_portfolio.py
│   ├── report.py
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
risk:
  initial_capital: 1000
  risk_per_trade: 25
  monthly_loss_limit: 100
  max_open_positions: 2
```

## Disclaimer

Il codice è didattico e sperimentale. Non costituisce consulenza finanziaria, raccomandazione personalizzata, sollecitazione all'investimento o garanzia di rendimento. Verifica sempre dati, costi, fiscalità, liquidità e idoneità dello strumento prima di operare con denaro reale.

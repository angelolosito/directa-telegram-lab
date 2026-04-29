# Quickstart

## 1. Crea repository GitHub privato

Nome consigliato: `directa-telegram-lab`.

## 2. Carica i file

Da terminale, dentro la cartella del progetto:

```bash
git init
git add .
git commit -m "Initial Directa Telegram Trading Lab"
git branch -M main
git remote add origin https://github.com/TUO_USERNAME/directa-telegram-lab.git
git push -u origin main
```

## 3. Crea il bot Telegram

- Apri Telegram.
- Cerca `@BotFather`.
- Scrivi `/newbot`.
- Copia il token.
- Scrivi un messaggio qualsiasi al nuovo bot.
- Recupera il chat id aprendo nel browser:

```text
https://api.telegram.org/bot<TOKEN>/getUpdates
```

## 4. Inserisci i secrets GitHub

Repository → Settings → Secrets and variables → Actions → New repository secret.

Crea:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

## 5. Test manuale

Repository → Actions → Daily Trading Alerts → Run workflow.

Dopo il primo run controlla:

- messaggio Telegram;
- file `reports/report_YYYY-MM-DD.md`;
- database `state/trading_lab.sqlite`;
- file `data/signals_log.csv`.

## 6. Regola watchlist e rischio

Modifica:

```text
watchlist.yaml
config.yaml
```

poi fai commit/push.

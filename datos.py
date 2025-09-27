import yfinance as yf

# Símbolo de Bitcoin en Yahoo Finance
ticker = "BTC-USD"

# =========================
# Datos diarios
# =========================
btc_daily = yf.download(ticker, period="max", interval="1d")
btc_daily.to_csv("bitcoin_diario.csv")
print("✅ Archivo diario guardado como bitcoin_diario.csv")

# =========================
# Datos semanales
# =========================
btc_weekly = yf.download(ticker, period="max", interval="1wk")
btc_weekly.to_csv("bitcoin_semanal.csv")
print("✅ Archivo semanal guardado como bitcoin_semanal.csv")

# =========================
# Datos mensuales
# =========================
btc_monthly = yf.download(ticker, period="max", interval="1mo")
btc_monthly.to_csv("bitcoin_mensual.csv")
print("✅ Archivo mensual guardado como bitcoin_mensual.csv")

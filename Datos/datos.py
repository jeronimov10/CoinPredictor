import yfinance as yf

#Imports foraneos
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ticker = "BTC-USD"

# =========================
# Datos diarios
# =========================
btc_daily = yf.download(ticker, period="max", interval="1d")
btc_daily.to_csv("bitcoin_diario.csv")


# =========================
# Datos semanales
# =========================
btc_weekly = yf.download(ticker, period="max", interval="1wk")
btc_weekly.to_csv("bitcoin_semanal.csv")


# =========================
# Datos mensuales
# =========================
btc_monthly = yf.download(ticker, period="max", interval="1mo")
btc_monthly.to_csv("bitcoin_mensual.csv")




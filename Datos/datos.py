import yfinance as yf
import pandas as pd


ticker = "BTC-USD"
def descargar_datos():
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






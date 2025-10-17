#Imports foraneos
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


    # =========================
    # Datos mensuales
    # =========================
    btc_monthly = yf.download(ticker, period="max", interval="1mo")
    btc_monthly.to_csv("bitcoin_mensual.csv")


def cargar_depurar_datos(archivo):
    """
    Carga y depura un archivo CSV de precios de Bitcoin mensual,
    devolviendo tanto el DataFrame limpio como una estructura
    tipo lista de diccionarios de diccionarios.
    """
    
    df = pd.read_csv(archivo, skiprows=2)

    
    df = df.rename(columns={
        "Date": "Date",
        "Unnamed: 1": "Close",
        "Unnamed: 2": "High",
        "Unnamed: 3": "Low",
        "Unnamed: 4": "Open",
        "Unnamed: 5": "Volume"
    })

    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

   
    df.dropna(inplace=True)

   
    df.set_index("Date", inplace=True)

    return df
archivo_semanal = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_semanal.csv"
archivo_diario = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_diario.csv"



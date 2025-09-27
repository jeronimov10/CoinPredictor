import pandas as pd

import matplotlib as plt

import mplfinance as mpf

import yfinance as yf

import numpy as np

archivo = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_semanal.csv"


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

    
    estructura = []
    for date, row in df.iterrows():
        estructura.append({
            str(date.date()): {
                "Close": row["Close"],
                "High": row["High"],
                "Low": row["Low"],
                "Open": row["Open"],
                "Volume": row["Volume"]
            }
        })

    return df, estructura


df, estrcutura = cargar_depurar_datos(archivo)

def graficas_ciclos(df)->None:
    """
    Genera gráficas de los ciclos identificados en los datos.
    """

    c1=df.loc["2012-11-28":"2016-07-09"]
    c2=df.loc["2016-07-09":"2020-05-11"]
    c3=df.loc["2020-05-11":"2024-04-19"]
    c4=df.loc["2024-04-19":"2028-04-10"]


    mpf.plot(c1, type='candle', style='charles',title='Ciclo 2012-2016', ylabel='Precio (USD)', volume=True, mav=(3,6,9))
    mpf.plot(c2, type='candle', style='charles',title='Ciclo 2016-2020', ylabel='Precio (USD)', volume=True, mav=(3,6,9))
    mpf.plot(c3, type='candle', style='charles',title='Ciclo 2020-2024', ylabel='Precio (USD)', volume=True, mav=(3,6,9))
    mpf.plot(c4, type='candle', style='charles',title='Ciclo 2024-2028 (en curso)', ylabel='Precio (USD)', volume=True, mav=(3,6,9))
    mpf.plot(df, type='candle', style='charles',title='Precio Bitcoin Diario', ylabel='Precio (USD)', volume=True, mav=(3,6,9))

    




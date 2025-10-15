import sys
import os
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulaciones.semanal import simulacion_montecarlo, cargar_depurar_datos
from NLP_Sentiment.NLP_main import news_api

archivo = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_semanal.csv"



def contexto_simulacion_montecarlo():
    df,estructura = cargar_depurar_datos(archivo)
    d, ds = simulacion_montecarlo(df)

    ds.describe()


    return 

    
    

def calficacion_sentimiento():
    s = news_api()
    return s

print(calficacion_sentimiento())


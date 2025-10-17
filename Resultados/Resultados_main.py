import sys, os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from Datos.datos import cargar_depurar_datos, archivo_semanal, archivo_diario
from Simulaciones.semanal import simulacion_montecarlo, multiples_simulaciones, simulacion_series_de_tiempo, grafica_simple
from Red_Neuronal.Red_Neuronal_Precios_BTC import df_30d
from NLP_Sentiment.NLP_main import news_api, yahoo_finance


d_semanal = cargar_depurar_datos(archivo_semanal)

simulacion_monte, df_m = simulacion_montecarlo(d_semanal)
df_c = multiples_simulaciones(d_semanal)
df_c = df_c.loc["2025-10-26":"2025-12-28"]
df_st, sim = simulacion_series_de_tiempo(d_semanal)

print(df_m.info())
print(df_c.info())
print(df_st.info())
print(df_30d.info())


grafica_simple(df_m)
grafica_simple(df_c)
grafica_simple(df_st)
grafica_simple(df_30d)
print(news_api())













from typing import List
import pandas as pd
from datetime import timedelta
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from math import ceil
import mplfinance as mpf
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import random
import warnings
warnings.filterwarnings('ignore')

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


#Dataframe depurado y una etsructura de datos tipo lista de diccionarios de diccionarios

df = cargar_depurar_datos(archivo_semanal = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_semanal.csv")

#Subsets de los ciclos del mercado teniendo en cuenta los halvings
c1=df.loc["2012-11-28":"2016-07-09"]
c2=df.loc["2016-07-09":"2020-05-11"]
c3=df.loc["2020-05-11":"2024-04-19"]
c4=df.loc["2024-04-19":"2028-04-10"]

def graficas_ciclos(c1, c2, c3, c4)->None:
    """
    Genera gráficas de los ciclos identificados en los datos.
    """

    fecha_max_1 = c1['Close'].idxmax()
    fecha_min_1 = c1['Close'].idxmin()    
    min_1 = float(c1.loc[fecha_min_1, 'Close'])
    max_1 = float(c1.loc[fecha_max_1, 'Close']) 
    prom_1 = float(c1['Close'].mean())

    fecha_max_2 = c2['Close'].idxmax()
    fecha_min_2 = c2['Close'].idxmin() 
    min_2 = float(c2.loc[fecha_min_2, 'Close'])
    max_2 = float(c2.loc[fecha_max_2, 'Close']) 
    prom_2 = float(c2['Close'].mean()) 
    
    fecha_max_3 = c3['Close'].idxmax()
    fecha_min_3 = c3['Close'].idxmin()  
    min_3 = float(c3.loc[fecha_min_3, 'Close'])
    max_3 = float(c3.loc[fecha_max_3, 'Close'])
    prom_3 = float(c3['Close'].mean())

    fecha_max_4 = c4['Close'].idxmax()
    fecha_min_4 = c4['Close'].idxmin() 
    prom_4 = float(c4['Close'].mean())
    min_4 = float(c4.loc[fecha_min_4, 'Close'])
    max_4 = float(c4.loc[fecha_max_4, 'Close'])  

    # --- Ciclo 2012-2016 ---
    fig, axlist = mpf.plot(
        c1, type='candle', style='charles',
        title='Ciclo 2012-2016', ylabel='Precio (USD)',
        volume=True, mav=(3,6,9), returnfig=True
    )
    ax = axlist[0]
    ax.text(
        0.02, 0.98,
        f"• Max Close: {max_1:,.0f} USD ({fecha_max_1:%Y-%m})\n"
        f"• Min Close: {min_1:,.0f} USD ({fecha_min_1:%Y-%m})\n"
        f"• Prom Close: {prom_1:,.0f} USD ({fecha_min_1:%Y-%m})",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
    )
    mpf.show()

        # --- Ciclo 2016-2020 ---
    fig, axlist = mpf.plot(
        c2, type='candle', style='charles',
        title='Ciclo 2016-2020', ylabel='Precio (USD)',
        volume=True, mav=(3,6,9), returnfig=True
    )
    ax = axlist[0]
    ax.text(
        0.02, 0.98,
        f"• Max Close: {max_2:,.0f} USD ({fecha_max_2:%Y-%m})\n"
        f"• Min Close: {min_2:,.0f} USD ({fecha_min_2:%Y-%m})\n"
        f"• Prom Close: {prom_2:,.0f} USD ({fecha_min_2:%Y-%m})",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
    )
    mpf.show()

    # --- Ciclo 2020-2024 ---
    fig, axlist = mpf.plot(
        c3, type='candle', style='charles',
        title='Ciclo 2020-2024', ylabel='Precio (USD)',
        volume=True, mav=(3,6,9), returnfig=True
    )
    ax = axlist[0]
    ax.text(
        0.02, 0.98,
        f"• Max Close: {max_3:,.0f} USD ({fecha_max_3:%Y-%m})\n"
        f"• Min Close: {min_3:,.0f} USD ({fecha_min_3:%Y-%m})\n"
        f"• Prom Close: {prom_3:,.0f} USD ({fecha_min_3:%Y-%m})",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
    )
    mpf.show()

    # --- Ciclo 2024-2028 (en curso) ---
    fig, axlist = mpf.plot(
        c4, type='candle', style='charles',
        title='Ciclo 2024-2028 (en curso)', ylabel='Precio (USD)',
        volume=True, mav=(3,6,9), returnfig=True
    )
    ax = axlist[0]
    ax.text(
        0.02, 0.98,
        f"• Max Close: {max_4:,.0f} USD ({fecha_max_4:%Y-%m})\n"
        f"• Min Close: {min_4:,.0f} USD ({fecha_min_4:%Y-%m})\n"
        f"• Prom Close: {prom_4:,.0f} USD ({fecha_min_4:%Y-%m})",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
    )
    mpf.show()

def grafica_simple(c)->None:
    """
    Genera una gráfica de un ciclo específico.
    """
    
    fecha_max = c['Close'].idxmax()
    fecha_min = c['Close'].idxmin()    
    min = float(c.loc[fecha_min, 'Close'])
    max = float(c.loc[fecha_max, 'Close']) 
    prom = float(c['Close'].mean())


    fig, axlist = mpf.plot(
        c, type='candle', style='charles',
        title='Ciclo 2024-2028 (en curso)', ylabel='Precio (USD)',
        volume=True, mav=(3,6,9), returnfig=True
    )
    ax = axlist[0]
    ax.text(
        0.02, 0.98,
        f"• Max Close: {max:,.0f} USD ({fecha_max:%Y-%m})\n"
        f"• Min Close: {min:,.0f} USD ({fecha_min:%Y-%m})\n"
        f"• Prom Close: {prom:,.0f} USD ({fecha_min:%Y-%m})",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
    )
    mpf.show()

def grafica_historica_con_ciclos(df, c1, c2, c3, c4)->None:
    """
    Genera una gráfica histórica completa de los datos.
    """
    
    fronteras = [
        "2012-11-28",  
        "2016-07-09",
        "2020-05-11",
        "2024-04-19",
    ]

    fecha_max_1 = c1['Close'].idxmax()
    fecha_min_1 = c1['Close'].idxmin()    
    min_1 = round(float(c1.loc[fecha_min_1, 'Close']),2)
    max_1 = round(float(c1.loc[fecha_max_1, 'Close']),2) 
    prom_1 = round(float(c1['Close'].mean()),2)

    fecha_max_2 = c2['Close'].idxmax()
    fecha_min_2 = c2['Close'].idxmin() 
    min_2 = round(float(c2.loc[fecha_min_2, 'Close']),2)
    max_2 = round(float(c2.loc[fecha_max_2, 'Close']),2) 
    prom_2 = round(float(c2['Close'].mean()),2) 
    
    fecha_max_3 = c3['Close'].idxmax()
    fecha_min_3 = c3['Close'].idxmin()  
    min_3 = round(float(c3.loc[fecha_min_3, 'Close']),2)
    max_3 = round(float(c3.loc[fecha_max_3, 'Close']),2)
    prom_3 = round(float(c3['Close'].mean()),2)

    fecha_max_4 = c4['Close'].idxmax()
    fecha_min_4 = c4['Close'].idxmin() 
    prom_4 = round(float(c4['Close'].mean()),2)
    min_4 = round(float(c4.loc[fecha_min_4, 'Close']),2)
    max_4 = round(float(c4.loc[fecha_max_4, 'Close']),2)  

    #Impresiones en consola datos de los ciclos
    print('El ciclo 1 abarco desde ' + str(c1.index[0].date()) + ' hasta ' + str(c1.index[-1].date()) + 'con el halving del ciclo siendo: 2012-11-28' )
    print('El ciclo 2 abarco desde ' + str(c2.index[0].date()) + ' hasta ' + str(c2.index[-1].date()) + 'con el halving del ciclo siendo: 2012-11-28')
    print('El ciclo 3 abarco desde ' + str(c3.index[0].date()) + ' hasta ' + str(c3.index[-1].date()) + 'con el halving del ciclo siendo: 2020-05-11')
    print('El ciclo 4 abarco desde ' + str(c4.index[0].date()) + ' hasta ' + str(c4.index[-1].date()) + 'con el halving del ciclo siendo: 2024-04-19' )

    print('El maximo en el ciclo 1 fue: ' + str(max_1) + ' USD el ' + str(fecha_max_1) + 'y el minimo fue: '+ str(min_1) + ' USD el ' + str(fecha_min_1) +  'el promedio fue: ' + str(prom_1) + ' USD')
    print('El maximo en el ciclo 2 fue: ' + str(max_2) + ' USD el ' + str(fecha_max_2) + 'y el minimo fue: '+ str(min_2) + ' USD el ' + str(fecha_min_2) +  'el promedio fue: ' + str(prom_2) + ' USD')
    print('El maximo en el ciclo 3 fue: ' + str(max_3) + ' USD el ' + str(fecha_max_3) + 'y el minimo fue: '+ str(min_3) + ' USD el ' + str(fecha_min_3) +  'el promedio fue: ' + str(prom_3) + ' USD')
    print('El maximo en el ciclo 4(Actual) fue: ' + str(max_4) + ' USD el ' + str(fecha_max_4) + 'y el minimo fue: '+ str(min_4) + ' USD el ' + str(fecha_min_4) +  'el promedio fue: ' + str(prom_4) + ' USD')




    #Grafica historica completa
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style='charles',
        volume=True,
        mav=(3, 6, 9),
        title='BTC Histórico — líneas de división por halving',
        ylabel='Precio (USD)',
        returnfig=True,
        yscale='log',
    )
    ax = axlist[0]

    
    posiciones = []
    for d in fronteras:
        ts = pd.Timestamp(d)
        pos = df.index.get_indexer([ts], method='nearest')[0]
        posiciones.append(pos)
        ax.axvline(pos, color='black', linewidth=1.6, alpha=0.95)

    


    mpf.show()

def fases_ciclo(c):

    """
    Determina las fases del ciclo dividiendola en 
    Fase alcista, Fase bajista, Recuperación (usando la pendiente y regersion lineal).
    Devuelve un DataFrame con las fases y el cambio porcentual semanal.
    """
    
    df = c.copy()
    df.index.name = "Date"

   
    ventana = 4
    tol_slope_pct = 0.005      
    rango_rec_pct = 0.08       
    tol_ret = 0.05             
    min_run = 5

    precios_cierre = df["Close"].astype(float).values
    logc = np.log(precios_cierre)


    rend = pd.Series(precios_cierre).pct_change().values
    vol_roll = pd.Series(rend).rolling(ventana).std().values

    fases = []
    slopes = []
    ranges = []

    X = np.arange(ventana).reshape(-1, 1)
    lr = LinearRegression()

    for i in range(len(precios_cierre)):
        if i < ventana:
            fases.append('Indefinido')
            slopes.append(np.nan)
            ranges.append(np.nan)
            continue

        
        y = logc[i-ventana:i]
        lr.fit(X, y)
        slope_log = lr.coef_[0]
        slope_pct = float(slope_log)   

        
        w = precios_cierre[i-ventana:i]
        rango_rel = (w.max() - w.min()) / w.mean()

        
        ret_net = float(logc[i-1] - logc[i-ventana])

        
        vol = vol_roll[i-1] if not np.isnan(vol_roll[i-1]) else None
        if vol is not None and vol > 0:
           
            tol_slope_dyn = max(tol_slope_pct, 0.5 * vol)          
            tol_ret_dyn   = max(tol_ret,       1.0 * vol)          
            rango_lat_dyn = max(rango_rec_pct, 1.2 * vol)          
        else:
            tol_slope_dyn = tol_slope_pct
            tol_ret_dyn   = tol_ret
            rango_lat_dyn = rango_rec_pct

        
       
        if (rango_rel <= rango_lat_dyn) and (abs(slope_pct) <= tol_slope_dyn):
            fase = 'Recuperacion'

        else:
            
            if (slope_pct >  tol_slope_dyn) and (ret_net >  tol_ret_dyn):
                fase = 'Alcista'
            elif (slope_pct < -tol_slope_dyn) and (ret_net < -tol_ret_dyn):
                fase = 'Bajista'
            else:
                
                if abs(slope_pct) > tol_slope_dyn and rango_rel > rango_lat_dyn:
                    fase = 'Alcista' if slope_pct > 0 else 'Bajista'
                else:
                    
                    fase = 'Recuperacion'

        fases.append(fase)
        slopes.append(slope_pct)
        ranges.append(rango_rel)

    
    fases_arr = np.array(fases, dtype=object)
    start = 0
    for i in range(1, len(fases_arr)+1):
        if i == len(fases_arr) or fases_arr[i] != fases_arr[i-1]:
            run_len = i - start
            if fases_arr[start] != 'Indefinido' and run_len < min_run:
                if start > 0:
                    fases_arr[start:i] = fases_arr[start-1]
                elif i < len(fases_arr):
                    fases_arr[start:i] = fases_arr[i]
            start = i

    pct = df["Close"].pct_change()

    out = pd.DataFrame({
        "Open": df["Open"],
        "High": df["High"],
        "Low": df["Low"],
        "Close": df["Close"],
        "Volume": df["Volume"],
        "Fase": fases_arr,
        "Pct_Change": pct
    })
    out.index.name = "Date"
    return out

def grafica_fases(c):

    """
    Genera una gráfica de las fases del ciclo actual con colores.
    1. Fase alcista: verde
    2. Fase bajista: rojo  
    3. Recuperación: azul
    4. Indefinido: gris
    """
    d = c.copy().sort_index()
    colores = {'Alcista': 'green', 'Bajista': 'red', 'Recuperación': 'blue', 'Indefinido': 'gray'}
    idx = d.index
    close = d['Close'].values
    fases = d['Fase'].values
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title('Ciclo 2024-2028 (en curso)')
    ax.set_ylabel('Precio (USD)')
    i0 = 0
    for i in range(1, len(d)):
        if fases[i] != fases[i-1]:
            ax.plot(idx[i0:i], close[i0:i], color=colores.get(fases[i-1], 'gray'), linewidth=2)
            i0 = i
    ax.plot(idx[i0:], close[i0:], color=colores.get(fases[-1], 'gray'), linewidth=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def estadisticas_dataframe(c):
    """
    Calcula estadísticas descriptivas del DataFrame.
    
    """

    periodos_anuales = 52
    df = c.copy()

    info_basica = {
        'fecha_inicio': df.index[0],
        'fecha_fin': df.index[-1],
        'num_periodos': len(df),
        'duracion_dias': (df.index[-1] - df.index[0]).days,
        'precio_inicial': df['Close'].iloc[0],
        'precio_final': df['Close'].iloc[-1],
        'precio_minimo': df['Close'].min(),
        'precio_maximo': df['Close'].max(),
        'volumen_promedio': df['Volume'].mean(),
        'volumen_total': df['Volume'].sum(),
    }

    returns = df['Close'].pct_change().dropna()
    
    

    volatilidad = {
        'volatilidad_periodo': returns.std() * 100,
        'volatilidad_anual': returns.std() * np.sqrt(periodos_anuales) * 100,
        'desviacion_estandar': returns.std(),
        'varianza': returns.var(),
        'rango_intercuartil': (returns.quantile(0.75) - returns.quantile(0.25)) * 100,
        'coef_variacion': (returns.std() / returns.mean()) if returns.mean() != 0 else np.nan,
    }




    resumen = {
        **info_basica,
        **volatilidad,
    }
    
    return resumen


#Simulacion Montecarlo ciclo actual esta simulacion de montecarlo es simple y no tiene en cuenta las fases del ciclo y solo se parctica como aporximado a posibles maixmos que vamos a encontarr sin embargo, no es bueno interpertarla por ciclos pues no tiene ciclos reales

def simulacion_montecarlo(c):
    """
    Simulación de Montecarlo avanzada para predecir precios futuros
    basada en el ciclo actual de Bitcoin.
    """
    simulaciones = 1000
    duracion_simulacion = 10 

    ultimo_precio = c['Close'].iloc[-1]
    ultima_fecha = c.index[-1]

    retornos = c['Close'].pct_change().dropna()

    mu = retornos.mean()
    sigma = retornos.std()

    close_to_open = (c['Open'] / c['Close'].shift(1)).dropna()
    high_to_close = (c['High'] / c['Close']).dropna()
    low_to_close  = (c['Low']  / c['Close']).dropna()
    
    gap_mean = close_to_open.mean()
    gap_std  = close_to_open.std()

    high_mean = high_to_close.mean()
    high_std  = high_to_close.std()

    low_mean = low_to_close.mean()
    low_std  = low_to_close.std()

    volumen_returns = c['Volume'].pct_change().dropna()
    volumen_mu = volumen_returns.mean()
    volumen_sigma = volumen_returns.std()
    ultimo_volumen = c['Volume'].iloc[-1]

    fechas_f = pd.date_range(start=ultima_fecha + timedelta(weeks=1), periods=duracion_simulacion, freq='W')
    
    
    opens_all   = np.zeros((simulaciones, duracion_simulacion))
    highs_all   = np.zeros((simulaciones, duracion_simulacion))
    lows_all    = np.zeros((simulaciones, duracion_simulacion))
    closes_all  = np.zeros((simulaciones, duracion_simulacion))
    volumes_all = np.zeros((simulaciones, duracion_simulacion))

    for sim in range(simulaciones):
        opens   = np.zeros(duracion_simulacion)
        highs   = np.zeros(duracion_simulacion)
        lows    = np.zeros(duracion_simulacion)
        closes  = np.zeros(duracion_simulacion)
        volumes = np.zeros(duracion_simulacion)

        current_close  = ultimo_precio
        current_volume = ultimo_volumen

        for semana in range(duracion_simulacion):

            # Simular Open
            gap_factor = np.random.normal(gap_mean, gap_std)
            gap_factor = max(0.8, min(1.2, gap_factor)) 
            opens[semana] = current_close * gap_factor

            # Simular Close
            random_return = np.random.normal(mu, sigma)
            new_close = current_close * (1 + random_return)
            closes[semana] = new_close
            
            # Simular High 
            high_factor = abs(np.random.normal(high_mean, high_std))
            high_factor = max(1.0, high_factor) 
            highs[semana] = new_close * high_factor
            highs[semana] = max(highs[semana], opens[semana], new_close)
            
            # Simular Low 
            low_factor = abs(np.random.normal(low_mean, low_std))
            low_factor = min(1.0, low_factor) 
            lows[semana] = new_close * low_factor
            lows[semana] = min(lows[semana], opens[semana], new_close)
            
            # Simular Volume
            volume_return = np.random.normal(volumen_mu, volumen_sigma)
            new_volume = current_volume * (1 + volume_return)
            new_volume = max(0, new_volume) 
            volumes[semana] = new_volume
            
            current_close  = new_close
            current_volume = new_volume

        opens_all[sim] = opens
        highs_all[sim] = highs
        lows_all[sim] = lows
        closes_all[sim] = closes
        volumes_all[sim] = volumes

   
    mean_open = opens_all.mean(axis=0)
    mean_high = highs_all.mean(axis=0)
    mean_low = lows_all.mean(axis=0)
    mean_close = closes_all.mean(axis=0)
    mean_volume = volumes_all.mean(axis=0)

    sim_df = pd.DataFrame({
        'Open':   mean_open,
        'High':   mean_high,
        'Low':    mean_low,
        'Close':  mean_close,
        'Volume': mean_volume
    }, index=fechas_f)

    
    std_open_prom = opens_all.std(axis=0).mean()
    std_high_prom = highs_all.std(axis=0).mean()
    std_low_prom = lows_all.std(axis=0).mean()
    std_close_prom = closes_all.std(axis=0).mean()
    std_volume_prom = volumes_all.std(axis=0).mean()

    
    print("Desviación estándar promedio (Open):",   std_open_prom)
    print("Desviación estándar promedio (High):",   std_high_prom)
    print("Desviación estándar promedio (Low):",    std_low_prom)
    print("Desviación estándar promedio (Close):",  std_close_prom)
    print("Desviación estándar promedio (Volume):", std_volume_prom)

    return pd.concat([c, sim_df]), sim_df


#Simulacion 2 series de tiempo ARIMA

def simulacion_series_de_tiempo(c):
    """
    Simulación de series de tiempo para predecir precios futuros
    Usa el modelo ARIMA
    basada en el ciclo historico del Bitcoin. Asumiendo que es una serie de
    tiempo estacionaria. Toca volverla no esta estacionaria.
    La tendencia historica del BTC ha sido a seguir patrones ciclicos
    Alcista, bajista y de recuperacion
    Sin embargo se ve que el BTC tiene una tendencia alcista a largo plazo
    y se espera que siga esa tendencia.
    Los ciclos alcistas se ven seguidos de ciclos bajistas seguido de ya sea un
    ciclo alcista o un ciclo de recupreacion o estabilizacion
    
    """
    periodos_futuros=12 
    seasonal_period=52

    d = c.copy().dropna()


    serie_close = d['Close'].copy()
    

    ultima_fecha = df.index[-1]
    fechas_futuras = pd.date_range(start = ultima_fecha + pd.Timedelta(weeks=1), periods=periodos_futuros, freq='W')

    #Implementacion modelando todas las columnas super lenta (no la he podido correr)

    # for columna in columnas:
    #     serie = d[columna].copy()

    #     modelo = pm.auto_arima(
    #         serie,
    #         seasonal=True,
    #         m=seasonal_period,
    #         start_p=0, start_q=0,
    #         max_p=3, max_q=3,
    #         start_P=0, start_Q=0,
    #         max_P=1, max_Q=1,
    #         d=None,
    #         D=1,
    #         trace=False,
    #         error_action='ignore',
    #         suppress_warnings=True,
    #         stepwise=True,
    #         information_criterion='aic',
    #         n_jobs=1, 
    #         maxiter=50
    #     )

    #     predicciones, _ = modelo.predict(n_periods=periodos_futuros, return_conf_int=True, alpha=0.05)

    #     predicciones_dict[columna] = predicciones

    # df_predicciones = pd.DataFrame(predicciones_dict, index=fechas_futuras)


    #Implementacion modelando solo el close y luego usando ratios para las demas columnas

    modelo = pm.auto_arima(
        serie_close.values,
        seasonal=False,  
        start_p=1, start_q=1,
        max_p=2, max_q=2,
        d=1,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        maxiter=50
    )



    pred_close = modelo.predict(n_periods=periodos_futuros)

    ratio_open = (d['Open'] / d['Close']).mean()
    ratio_high = (d['High'] / d['Close']).mean()
    ratio_low = (d['Low'] / d['Close']).mean()
    ratio_volume = (d['Volume'] / d['Close']).mean()

    df_predicciones = pd.DataFrame({
        'Open': pred_close * ratio_open,
        'High': pred_close * ratio_high,
        'Low': pred_close * ratio_low,
        'Close': pred_close,
        'Volume': pred_close * ratio_volume
    }, index=fechas_futuras)

    return df_predicciones.dropna(), pd.concat([d, df_predicciones])




#Simulacion 3 robusta cadenas de markov

def calcular_estadisticas_duracion_fases(c):
    """
    
    """

    d = c.copy()
    


    estadisticas = {}
    fases = d['Fase'].unique()
    
    for fase in fases:
        if fase == 'Indefinido':
            continue
        
        duraciones = []
        contador = 0
        
        for i in range(len(df)):
            if d['Fase'].iloc[i] == fase:
                contador += 1
            else:
                if contador > 0:
                    duraciones.append(contador)
                    contador = 0
        
        if contador > 0:
            duraciones.append(contador)
        
        if len(duraciones) > 0:
            estadisticas[fase] = {
                'promedio': np.mean(duraciones),
                'std': np.std(duraciones),
                'max': np.max(duraciones),
                'min': np.min(duraciones)
            }
        else:
            print('No fue posible obtner las estadisticas de las fases.')
    return estadisticas

def calcular_estadisticas_cambio_precio(c):

    """
    
    
    """

    d = c.copy()
    
    fases_activas = [f for f in d['Fase'].unique() if f != 'Indefinido']

    estadisticas_cambio = {}
   

    for fase in fases_activas:
        df_fase = d[d['Fase'] == fase].copy()


        if len(df_fase) > 1:

            cambios = df_fase['Pct_Change'].dropna()

            ratio_high = (df_fase['High'] / df_fase['Close']).mean()
            ratio_low = (df_fase['Low'] / df_fase['Close']).mean()
            ratio_open = (df_fase['Open'] / df_fase['Close']).mean()
            ratio_volume = (df_fase['Volume'] / df_fase['Close']).mean()

            estadisticas_cambio[fase] = {
                'cambio_promedio': cambios.mean(),
                'cambio_std': cambios.std(),
                'ratio_high': ratio_high,
                'ratio_low': ratio_low,
                'ratio_open': ratio_open,
                'ratio_volume': ratio_volume
            }


    return estadisticas_cambio


def calculo_probabilidades_cambio_fase(c):


    d = c.copy()
    d = fases_ciclo(d)
    fases = d['Fase'].unique()
        
    
    transiciones = {}
    for fase in fases:
        dic_i = {}
        for f in fases:
            dic_i[f] = 0
        transiciones[fase] = dic_i
        
    #cambios fases
    for i in range(len(d)-1):
        fase_actual = d['Fase'].iloc[i]
        fase_siguiente = d['Fase'].iloc[i+1]
        transiciones[fase_actual][fase_siguiente] += 1
            
    
    prob_transicion = pd.DataFrame(transiciones)
        
    #Normalizamos   
    for fase in fases:
        total = prob_transicion[fase].sum()
        if total > 0:
            prob_transicion[fase] = prob_transicion[fase] / total
                
    return prob_transicion


def simulacion_cadenas_markov(c):

    """
        Simulacion usando las cadenas de markov.
        Pseudocodigo de la implementacion en el readme
        Esta simulacion principalmente intenta 
        
    
    """


    d = fases_ciclo(c.copy())

    

    estadisticas_cambio = calcular_estadisticas_cambio_precio(d)
    estadisticas_duracion = calcular_estadisticas_duracion_fases(d)

    matriz_probabilidades = calculo_probabilidades_cambio_fase(d)

    num_semanas = 10

    k = 1.5

    fases_activas = [f for f in d['Fase'].unique() if f != 'Indefinido'] #Eliminar la fase indefinido


    #Quitar indefinido de la matriz
    if 'Indefinido' in matriz_probabilidades.columns:
        matriz_filtrada = matriz_probabilidades[fases_activas].loc[fases_activas]
    else:
        matriz_filtrada = matriz_probabilidades

    ultima_fase = d['Fase'].iloc[-1]

    fases_simuladas = []
    fase_actual = ultima_fase
    contador_fase = 0

    #Simular fase
    for semana in range(num_semanas):
        contador_fase += 1

        if fase_actual in estadisticas_duracion:
            duracion_max = (estadisticas_duracion[fase_actual]['promedio'] + k * estadisticas_duracion[fase_actual]['std'])


        if contador_fase < duracion_max:
            probabilidades = matriz_filtrada[fase_actual].values
            fases = matriz_filtrada.index.tolist()
            fase_siguiente = np.random.choice(fases, p=probabilidades)
        else:
            probabilidades = matriz_filtrada[fase_actual].copy()
            probabilidades[fase_actual] = 0


            if probabilidades.sum() > 0:
                probabilidades = probabilidades / probabilidades.sum()
                fases = matriz_filtrada.index.tolist()
                fase_siguiente = np.random.choice(fases, p=probabilidades.values)
            else:
                fases_disponibles = [f for f in fases_activas if f != fase_actual]
                fase_siguiente = np.random.choice(fases_disponibles)

        if fase_siguiente != fase_actual:
            contador_fase = 0

        fases_simuladas.append(fase_siguiente)
        fase_actual = fase_siguiente

    # Simular precios OHLCV
    ultimo_close = d['Close'].iloc[-1]
    
    precios_open = []
    precios_high = []
    precios_low = []
    precios_close = []
    volumenes = []
    
    precio_close_actual = ultimo_close
    
    for fase in fases_simuladas:
        stats = estadisticas_cambio[fase]
        
        
        cambio_porcentual = np.random.normal(stats['cambio_promedio'], stats['cambio_std'])
        
        
        precio_close = precio_close_actual * (1 + cambio_porcentual)
        
        
        precio_open = precio_close * stats['ratio_open']
        precio_high = precio_close * stats['ratio_high']
        precio_low = precio_close * stats['ratio_low']
        volumen = precio_close * stats['ratio_volume']
        
        
        precio_high = max(precio_high, precio_open, precio_close)
        precio_low = min(precio_low, precio_open, precio_close)
        
        
        precios_open.append(precio_open)
        precios_high.append(precio_high)
        precios_low.append(precio_low)
        precios_close.append(precio_close)
        volumenes.append(volumen)
        
        
        precio_close_actual = precio_close
    
    
    
    ultima_fecha = d.index[-1]
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(weeks=1),periods=num_semanas,freq='W')
    
    df_simulado = pd.DataFrame({
        'Open': precios_open,
        'High': precios_high,
        'Low': precios_low,
        'Close': precios_close,
        'Volume': volumenes
    }, index=fechas_futuras)
    
    return df_simulado


def multiples_simulaciones(c):

    """"
    
    
    """

    d = c.copy()
    num_simulaciones = 100
    todas_simulaciones = []
    #Simula n veces
    for i in range(num_simulaciones):
        df_sim = simulacion_cadenas_markov(d)
        todas_simulaciones.append(df_sim)

    #Construye el dataframe
    df_resultado = todas_simulaciones[0].copy()
    num_semanas = len(df_resultado)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        valores_promedio = []
        
        for semana_idx in range(num_semanas):
            valores_semana = [sim.iloc[semana_idx][col] for sim in todas_simulaciones]
            promedio = np.mean(valores_semana)
            valores_promedio.append(promedio)
        
        df_resultado[col] = valores_promedio



    return df_resultado


#Pruebas de las funciones




# c4_s, c4_s_s = simulacion_montecarlo(df)

# print(estadisticas_dataframe(c4_s_s))

# grafica_simple(c4_s_s)

# print(c4_s_s.info())

# c4_s, c4_s_s = simulacion_series_de_tiempo(df)
# grafica_simple(c4_s)
# estadisticas_dataframe(c4_s)

# fases_ciclo(df)
# grafica_un_ciclo(df)
# grafica_historica(df, c1, c2, c3, c4)
# grafica_fases(fases_ciclo(df))

# l, a = simulacion_series_de_tiempo(df)


# print(l.info())
# print(a.info())


# grafica_simple(l)
# grafica_simple(a)

# b = calculo_probabilidades_cambio_fase(c3)
# print("Indefinido")
# print(b['Indefinido'])
# print("Alcista")
# print(b['Alcista'])
# print("Bajista")
# print(b['Bajista'])
# print("Recuperacion")
# print(b['Recuperacion'])


# d = multiples_simulaciones(df)
# print(d.info())

# grafica_simple(d)

# fases = calculo_probabilidades_cambio_fase(df).index.tolist()


# probabilidades = calculo_probabilidades_cambio_fase(df)['Alcista'].values

# fase_siguiente = np.random.choice(fases, p=probabilidades)

# print(fase_siguiente)

# d,dd = simulacion_series_de_tiempo(df)

# grafica_simple(d)
# print(d.info())

# l = multiples_simulaciones(df)
# grafica_simple(l)
# print(l.info())



    















import pandas as pd

import matplotlib.pyplot as plt

import mplfinance as mpf

from sklearn.linear_model import LinearRegression

import yfinance as yf

import numpy as np


archivo = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_mensual.csv"


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


#Dataframe depurado y una etsructura de datos tipo lista de diccionarios de diccionarios
df, estrcutura = cargar_depurar_datos(archivo)

#Subsets de los ciclos del mercado teniendo en cuenta los halvings
c1=df.loc["2012-11-28":"2016-07-09"]
c2=df.loc["2016-07-09":"2020-05-11"]
c3=df.loc["2020-05-11":"2024-04-19"]
c4=df.loc["2024-04-19":"2028-04-10"]

def graficas_ciclos(c1, c2, c3,c4)->None:
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

def grafica_historica(df, c1,c2,c3,c4)->None:
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
    tol_slope_pct = 0.02
    tol_ret = 0.20
    rango_rec_pct = 0.10
    min_run = 3

    precios_cierre = df["Close"].astype(float).values
    logc = np.log(precios_cierre)

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

        if rango_rel <= rango_rec_pct and abs(slope_pct) <= tol_slope_pct:
            fase = 'Recuperación'
        elif (slope_pct >  tol_slope_pct) and (ret_net >  tol_ret):
            fase = 'Alcista'
        elif (slope_pct < -tol_slope_pct) and (ret_net < -tol_ret):
            fase = 'Bajista'
        else:
            fase = 'Recuperación'

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



import pandas as pd

import matplotlib as plt

import mplfinance as mpf

import yfinance as yf

import numpy as np

archivo = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_diario.csv"


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


def graficas_ciclos(c1,c2,c3,c4)->None:
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


def fases_ciclo(c, post_w: int = 52, pre_w: int = 36):
    """
    Determina las fases del ciclo dividiendola en 
    Bull point, Bear point, recuperacion, post Halving, pre Halving y
    Determina si el mercado esta en una fase alcista o esta en una fase bajista.
    """


    df = c.copy()
    df.index.name = "Date"

    inicio, final = df.index.min(), df.index.max()
    fecha_max_close = df["Close"].idxmax()
    post_max = df.loc[fecha_max_close:final]

    if not post_max.empty:
        fecha_min = post_max["Close"].idxmin()
    else:
        fecha_min = df["Close"].idxmin()

    #definir post y pre halving
    post_end = inicio + pd.to_timedelta(post_w, unit="W")
    if post_end > final:
        post_end = final

    pre_start = final - pd.to_timedelta(pre_w, unit="W")
    if pre_start < inicio:
        pre_start = inicio



    fase = pd.Series(index=df.index, dtype=object)
    fase[:] = ""


    # Bull point: inicio -> top
    left = min(inicio, fecha_max_close)
    right = max(inicio, fecha_max_close)
    fase.loc[left:right] = "Bull point"

    # Bear point: top -> bottom
    if fecha_min >= fecha_max_close:
        fase.loc[fecha_max_close:fecha_min] = "Bear point"

    # Recuperación: bottom -> pre_halving
    if fecha_min < pre_start:
        fase.loc[fecha_min:pre_start] = "Recuperación"

    # Post-halving:
    fase.loc[inicio:post_end] = "Post-halving"

    # Pre-halving:
    fase.loc[pre_start:final] = "Pre-halving"

    pct = df["Close"].pct_change()


    out = pd.DataFrame({
        "Open"       : df["Open"],
        "High"       : df["High"],
        "Low"        : df["Low"],
        "Close"      : df["Close"],
        "Volume"     : df["Volume"],
        "Fase"       : fase,
        "Pct_Change" : pct
    })
    out.index.name = "Date"
    



    return out

    






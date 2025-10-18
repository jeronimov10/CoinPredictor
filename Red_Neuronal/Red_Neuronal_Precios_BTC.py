import keras
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

from datetime import timedelta

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

df = cargar_depurar_datos(archivo_diario = "C:/Users/jeron/OneDrive/Escritorio/CoinPredictor/Datos/bitcoin_diario.csv")


def red_LSTM():
    

    set_entrenamiento = df.loc[:"2023-12-31"].iloc[:,1:2]
    set_validacion = df.loc["2024-01-01":].iloc[:,1:2]

    scaler = MinMaxScaler(feature_range=(0,1))


    set_entrenamiento_escalado = scaler.fit_transform(set_entrenamiento)

    
    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)


    for i in range(time_step,m):
        # X: bloques
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    neuronas = 150

    modelo = Sequential()


    #Capa 1 inicial - entrada
    modelo.add(LSTM(units=neuronas, input_shape=dim_entrada))

    #Capa salida
    modelo.add(Dense(units=dim_salida))


    print('Comenzando el entrenamiento...')
    modelo.compile(optimizer='rmsprop', loss='mse')

    modelo.fit(X_train,Y_train,epochs=50,batch_size=32)

    print('Entrenamiento completado exitosamente')

    #test y predicicones
    x_test = set_validacion.values
    x_test_scaled = scaler.transform(x_test)

    X_test = []
    for i in range(time_step,len(x_test_scaled)):
        X_test.append(x_test_scaled[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = scaler.inverse_transform(prediccion)

    p = np.mean(prediccion)
    t = np.mean(x_test)

    fechas_validacion = set_validacion.index[time_step:]
    valores_reales = set_validacion.iloc[time_step:, 0].values

    #Accuracy
    ac = 0
    if p > t:
        ac = round((t/p)*100,2)
    else:
        ac = round((p/t)*100,2)

    print('La exactitud es: '+ str(ac) + '%')

    #Rentabilidad
    rentability = 1
    for i in range(1,len(valores_reales)):
        if prediccion[i] > valores_reales[i-1]:
            rentability*= valores_reales[i]/valores_reales[i-1]

    print('La rentabilidad del modelo es: ' + str((rentability-1)*100)+"%")

    #Rentabilidad diaria
    daily_return = (rentability ** (1/len(valores_reales))) - 1
    print('La rentabilidad diaria del modelo es: ' + str(daily_return * 100) + "% por día")



    #Predicciones siguientes  dias

    dias_futuros = 20

    ultimo_bloque = df['High'].values[-time_step:]
    ultimo_bloque_escalado = scaler.transform(ultimo_bloque.reshape(-1, 1))

    predicciones_futuras = []

    bloque_actual = ultimo_bloque_escalado.copy()

    for i in range(dias_futuros):
        X_futuro = bloque_actual.reshape(1, time_step, 1)
        prediccion_escalada = modelo.predict(X_futuro, verbose=0)
        predicciones_futuras.append(prediccion_escalada[0, 0])
        bloque_actual = np.append(bloque_actual[1:], prediccion_escalada)

    predicciones_futuras = np.array(predicciones_futuras).reshape(-1, 1)
    predicciones_futuras = scaler.inverse_transform(predicciones_futuras)

    
    ultima_fecha = df.index[-1]
    fechas_futuras = [ultima_fecha + timedelta(days=i+1) for i in range(dias_futuros)]

    print("PREDICCIONES PARA LOS PRÓXIMOS", dias_futuros, "DÍAS:")
    print("="*60)
    for fecha, precio in zip(fechas_futuras, predicciones_futuras):
        print(f"{fecha.strftime('%Y-%m-%d')}: ${precio[0]:,.2f}")
    print("="*60)

    #Grafica
 
    plt.figure(figsize=(12, 6))
    plt.plot(valores_reales, label='Valor real de la acción', color='blue', linewidth=1.5)
    plt.plot(prediccion, label='Predicción de la acción', color='red', linewidth=1.5)

    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Valor de la acción', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


    return modelo



def predecir_ohlcv_30d(modelo, df, scaler=None, time_step=60, dias_futuros=30, k_rel=30, col_ref='High'):
    """
    Genera un DataFrame OHLCV con predicciones para los próximos `dias_futuros` días.
    - modelo: modelo LSTM ya entrenado (como el que devuelves en red_LSTM()).
    - df: DataFrame histórico con índice de fechas y columnas: ['Open','High','Low','Close','Volume'].
    - scaler: MinMaxScaler usado en el entrenamiento (ideal). Si es None, se ajusta sobre df[col_ref].
    - time_step: ventana usada en el entrenamiento (tu código usa 60).
    - dias_futuros: horizonte de predicción (por defecto 30).
    - k_rel: ventana (días recientes) para estimar relaciones OHLC con respecto a High.
    - col_ref: columna objetivo usada en el LSTM (en tu código: 'High').

    Devuelve: DataFrame con índice de fechas futuras y columnas ['Open','High','Low','Close','Volume'].
    """

    
    req_cols = {'Open','High','Low','Close','Volume'}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener columnas {req_cols}. Encontradas: {df.columns.tolist()}")

    if len(df) < time_step + 5:
        raise ValueError("df no tiene suficientes filas para construir el último bloque de predicción.")

    
    freq = pd.infer_freq(df.index)
    if freq is None:
        
        freq = 'D'

   
    fitted_locally = False
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0,1))
        
        scaler.fit(df[col_ref].values.reshape(-1,1))
        fitted_locally = True  

    
    ultimo_bloque = df[col_ref].values[-time_step:]
    ultimo_bloque_escalado = scaler.transform(ultimo_bloque.reshape(-1, 1))

    
    pred_high_scaled = []
    bloque_actual = ultimo_bloque_escalado.copy()
    for _ in range(dias_futuros):
        X_futuro = bloque_actual.reshape(1, time_step, 1)
        pred_esc = modelo.predict(X_futuro, verbose=0)
        pred_high_scaled.append(pred_esc[0,0])
        
        bloque_actual = np.append(bloque_actual[1:], pred_esc).reshape(-1,1)

    pred_high = scaler.inverse_transform(np.array(pred_high_scaled).reshape(-1,1)).ravel()

    #
    tail = df.iloc[-k_rel:] if len(df) >= k_rel else df
    
    eps = 1e-9
    r_close = np.median((tail['Close'] / (tail['High'] + eps)).clip(0.7, 1.0))  
    r_low   = np.median((tail['Low']   / (tail['High'] + eps)).clip(0.5, 0.99)) 
 
   
    vol_base = float(np.median(tail['Volume'])) if tail['Volume'].notna().any() else 0.0

    
    futuros_idx = pd.date_range(start=df.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                periods=dias_futuros, freq=freq)

    ohlcv_rows = []
    prev_close = float(df['Close'].iloc[-1])

    for i in range(dias_futuros):
        high_i = float(pred_high[i])

       
        close_i = float(max(eps, r_close * high_i))

        
        open_i = float(prev_close) if i > 0 else float(df['Close'].iloc[-1])

       
        low_guess = float(max(eps, r_low * high_i))
        low_i = min(low_guess, open_i, close_i, high_i)

        
        high_i = max(high_i, open_i, close_i, low_i)  
        low_i  = min(low_i, open_i, close_i, high_i)

        
        vol_i = vol_base

        ohlcv_rows.append([open_i, high_i, low_i, close_i, vol_i])
        prev_close = close_i

    df_futuro = pd.DataFrame(ohlcv_rows, index=futuros_idx, columns=['Open','High','Low','Close','Volume'])

    
    df_futuro['High'] = df_futuro[['Open','Close','High']].max(axis=1)
    df_futuro['Low']  = df_futuro[['Open','Close','Low']].min(axis=1)

    
    for c in ['Open','High','Low','Close']:
        df_futuro[c] = df_futuro[c].astype(float)

    return df_futuro


modelo = red_LSTM()
fecha_hoy = pd.Timestamp.now().strftime("%Y%m%d")
modelo.save(f'Modelo_LSTM{fecha_hoy}.h5')

#Cargar modelo
# model_b = keras.models.load_model(f'Modelo_LSTM{fecha_hoy}.h5')
df_30d = predecir_ohlcv_30d(modelo, df, scaler=None, time_step=60, dias_futuros=30)




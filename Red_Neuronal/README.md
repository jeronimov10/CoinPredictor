# Red Neuronal LSTM para Prediccion de Bitcoin

Este notebook implementa una red neuronal LSTM (Long Short-Term Memory) para predecir el precio de Bitcoin utilizando datos historicos.

## Descripcion General

El modelo utiliza datos diarios de Bitcoin desde septiembre de 2014 hasta octubre de 2025 para entrenar una red LSTM que predice el precio maximo (High) del siguiente dia basandose en los ultimos 60 dias de datos.

## Estructura del Notebook

### 1. Carga y Preprocesamiento de Datos

- **Fuente de datos**: `Datos/bitcoin_diario.csv`
- **Columnas utilizadas**: Date, Close, High, Low, Open, Volume
- **Division de datos**:
  - Entrenamiento: 2014-09-17 a 2023-12-31 (3,393 registros)
  - Validacion: 2024-01-01 a 2025-10-25 (664 registros)

### 2. Normalizacion

Se utiliza `MinMaxScaler` de scikit-learn para escalar los datos al rango [0, 1].

### 3. Preparacion de Secuencias

- **Time step**: 60 dias
- Se crean secuencias donde X contiene los ultimos 60 dias y Y el valor del dia siguiente
- Los datos se reformatean a 3D para compatibilidad con LSTM: `(samples, time_steps, features)`

### 4. Arquitectura del Modelo

```
Sequential:
  - LSTM (150 neuronas, input_shape=(60, 1))
  - Dense (1 neurona, salida)
```

**Parametros de entrenamiento**:
- Optimizador: RMSprop
- Funcion de perdida: MSE (Mean Squared Error)
- Epochs: 50
- Batch size: 32

### 5. Evaluacion del Modelo

El notebook calcula varias metricas de rendimiento:

| Metrica | Valor |
|---------|-------|
| Precision promedio | 95.31% |
| Rentabilidad acumulada | 101.48% |
| Precision direccional | 49.97% |
| Retorno diario promedio | 0.116% |

### 6. Prediccion Futura

El modelo genera predicciones para los proximos 10 dias utilizando un enfoque iterativo donde cada prediccion se incorpora al bloque de entrada para la siguiente prediccion.

## Requisitos

```python
numpy
pandas
matplotlib
scikit-learn
keras / tensorflow
```

## Uso

1. Asegurate de tener el archivo CSV de datos en la ruta especificada
2. Ejecuta todas las celdas del notebook en orden
3. El modelo se entrenara y mostrara:
   - Graficos comparativos de valores reales vs predicciones
   - Metricas de rendimiento
   - Predicciones para los proximos 10 dias

## Notas

- El modelo utiliza una semilla aleatoria (`np.random.seed(4)`) para reproducibilidad
- Las predicciones futuras tienden a mostrar una tendencia debido a la naturaleza autorregresiva del enfoque
- La precision direccional cercana al 50% indica que el modelo tiene dificultades para predecir la direccion del movimiento del precio

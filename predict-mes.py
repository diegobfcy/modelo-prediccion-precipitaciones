# predict.py

import numpy as np
import tensorflow as tf
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta # Necesitarás instalar: pip install python-dateutil
import matplotlib.pyplot as plt
import os
import joblib # Para cargar el scaler si se usó
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker # Importar para control de ticks

from config import (
    DATA_PATH, PREC_VARIABLE_NAME, TIME_DIM, Y_DIM, X_DIM,
    SEQUENCE_LENGTH, MODEL_SAVE_PATH, PROCESSED_DATA_DIR
)

def get_last_date_in_dataset(ds_time_dim_len):
    """
    Calcula la última fecha en el conjunto de datos basándose en la longitud
    de la dimensión de tiempo y un punto de inicio fijo (Ene 1981).
    El dataset va de Ene 1981 a Feb 2025 (530 pasos si ds_time_dim_len es 530).
    """
    start_year = 1981
    start_month = 1
    # ds_time_dim_len debe ser 530 para tu dataset (Ene 1981 es el paso 0)
    total_months_offset = ds_time_dim_len - 1 # Meses desde el inicio (0-indexado)
    
    start_date = datetime(start_year, start_month, 1)
    end_date = start_date + relativedelta(months=total_months_offset)
    return end_date

# --- Funciones formateadoras para los ticks de los ejes ---
# Estas funciones asumen que reciben el valor *geográfico* del tick
def lat_formatter(tick_val, pos):
    """ Formatea los ticks de latitud. Ej: 0.95N, 18.75S """
    # Asegurarse de que tick_val es un número flotante
    tick_val = float(tick_val)
    if tick_val > 0: # Norte
        return f'{tick_val:.2f}N'
    elif tick_val < 0: # Sur (el valor es negativo)
        return f'{abs(tick_val):.2f}S'
    else: # Ecuador
        return '0°'

def lon_formatter(tick_val, pos):
    """ Formatea los ticks de longitud. Ej: 81.3W, 68W """
    # Asegurarse de que tick_val es un número flotante
    tick_val = float(tick_val)
    if tick_val < 0: # Oeste (el valor es negativo)
        val_abs = abs(tick_val)
        # Formato con 0 decimales si es un entero, 1 decimal si no lo es
        if np.isclose(val_abs, round(val_abs), atol=1e-2): # Usar atol por seguridad con floats
            formatted_val = f'{val_abs:.0f}' # Para "68W"
        else:
            formatted_val = f'{val_abs:.1f}' # Para "81.3W"
        return f'{formatted_val}W'
    elif tick_val > 0: # Este
         val_abs = abs(tick_val)
         if np.isclose(val_abs, round(val_abs), atol=1e-2):
            formatted_val = f'{val_abs:.0f}'
         else:
            formatted_val = f'{val_abs:.1f}'
         return f'{formatted_val}E'
    else: # Meridiano Cero
        return '0°'

def main():
    # --- Entrada del Usuario para la Fecha de Predicción Objetivo ---
    try:
        target_year_str = input("Ingresa el año objetivo para la predicción (ej. 2025): ")
        target_month_str = input("Ingresa el mes objetivo para la predicción (ej. 3 para Marzo, 12 para Diciembre): ")
        target_year = int(target_year_str)
        target_month = int(target_month_str)
        if not (1 <= target_month <= 12):
            raise ValueError("El mes debe estar entre 1 y 12.")
        target_pred_date = datetime(target_year, target_month, 1)
    except ValueError as e:
        print(f"Entrada de fecha inválida: {e}")
        return

    print(f"Cargando datos desde: {DATA_PATH}")
    ds = None # Inicializar ds a None
    try:
        # Cargar el dataset completo para obtener datos y coordenadas
        # !!! IMPORTANTE: Usar decode_times=False aquí también !!!
        ds = xr.open_dataset(DATA_PATH, decode_times=False)
        
        num_total_data_steps = len(ds[TIME_DIM]) # Debería ser 530
        
        # Obtener las coordenadas de latitud y longitud
        # Asegurarse de que las coordenadas existen y tienen la dimensión correcta
        if Y_DIM not in ds.coords or X_DIM not in ds.coords:
             raise ValueError(f"Las dimensiones de coordenada '{Y_DIM}' o '{X_DIM}' no se encontraron en el dataset.")

        lat_coords = ds[Y_DIM].values # Array de valores de latitud
        lon_coords = ds[X_DIM].values # Array de valores de longitud

        # Verificar la correspondencia entre las dimensiones del array de datos y las coordenadas
        # Asumiendo que la variable de precipitación tiene dimensiones (Tiempo, Y, X) después del transpose
        precipitation_data_raw = ds[PREC_VARIABLE_NAME].transpose(TIME_DIM, Y_DIM, X_DIM).values
        
        y_dim_size = precipitation_data_raw.shape[1]
        x_dim_size = precipitation_data_raw.shape[2]

        if len(lat_coords) != y_dim_size or len(lon_coords) != x_dim_size:
             raise ValueError(f"Las dimensiones de los datos de precipitación ({y_dim_size}, {x_dim_size}) no coinciden con las dimensiones de las coordenadas ({len(lat_coords)}, {len(lon_coords)}).")

        # Verificar el orden de las latitudes si 'origin="upper"' está configurado
        # Si el primer valor de lat_coords es el más grande (más al norte) y el último es el más pequeño (más al sur),
        # entonces la correspondencia con 'origin="upper"' es correcta.
        # Para este caso específico (0.95N a 18.75S), las latitudes deberían ir decreciendo (positivas -> cero -> negativas).
        # Si lat_coords[0] > lat_coords[-1], el orden es correcto para origin='upper'.
        # print(f"Latitudes: {lat_coords[0]} a {lat_coords[-1]}") # Para depuración
        if not (lat_coords[0] >= lat_coords[-1]):
             print("Advertencia: Las coordenadas de latitud no parecen estar en orden decreciente (Norte a Sur). El etiquetado podría no ser correcto con 'origin=\"upper\"'.")
             # Considerar invertir lat_coords o ajustar la lógica de ticks si es necesario.
             # Para los fines de este ejemplo, asumiremos que el orden decreciente es el esperado.

        # Verificar el orden de las longitudes (81.3W a 68W). Estas son longitudes negativas.
        # Los valores deberían ir de -81.3 a -68 (orden creciente).
        # Si lon_coords[0] < lon_coords[-1], el orden es correcto.
        # print(f"Longitudes: {lon_coords[0]} a {lon_coords[-1]}") # Para depuración
        if not (lon_coords[0] <= lon_coords[-1]):
             print("Advertencia: Las coordenadas de longitud no parecen estar en orden creciente (Oeste a Este, usando valores negativos). El etiquetado podría no ser correcto.")
             # Considerar invertir lon_coords o ajustar la lógica de ticks si es necesario.
             # Para los fines de este ejemplo, asumiremos que el orden creciente es el esperado.


        last_data_date = get_last_date_in_dataset(num_total_data_steps)
        print(f"El dataset contiene datos desde Ene 1981 hasta {last_data_date.strftime('%b %Y')}.")

        if target_pred_date <= last_data_date:
            print(f"La fecha objetivo ({target_pred_date.strftime('%b %Y')}) está dentro o antes del final del dataset ({last_data_date.strftime('%b %Y')}).")
            print("Este script está diseñado para predicciones futuras. Por favor, ingresa una fecha posterior a Febrero 2025.")
            ds.close()
            return

        # Calcular el número de meses a predecir hacia el futuro
        delta = relativedelta(target_pred_date, last_data_date)
        num_steps_to_predict = delta.years * 12 + delta.months

        if num_steps_to_predict <= 0:
             print(f"La fecha objetivo {target_pred_date.strftime('%b %Y')} no está en el futuro respecto al final del dataset {last_data_date.strftime('%b %Y')}.")
             ds.close()
             return
        
        print(f"Se predecirán {num_steps_to_predict} mes(es) hacia adelante para alcanzar {target_pred_date.strftime('%b %Y')}.")

        # precipitation_data_raw ya está cargado y reordenado arriba


        # --- Preprocesamiento de los datos históricos ---
        # 1. Manejar NaNs (igual que en data_preprocessing.py)
        if np.isnan(precipitation_data_raw).any():
            print("Advertencia: Se encontraron valores NaN en el dataset completo. Rellenando con 0.")
            precipitation_data_processed = np.nan_to_num(precipitation_data_raw, nan=0.0)
        else:
            precipitation_data_processed = precipitation_data_raw.copy() # Usar una copia para no modificar el original

        # 2. Opcional: Escalado (si se usó un scaler durante el entrenamiento)
        scaler = None
        scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                print("Scaler cargado. Aplicando a los datos.")
                original_shape = precipitation_data_processed.shape
                # El scaler espera (n_samples, n_features), es decir (T*Y*X, 1)
                precipitation_data_processed = scaler.transform(precipitation_data_processed.reshape(-1, 1)).reshape(original_shape)
            except Exception as e:
                print(f"Advertencia: No se pudo cargar o aplicar el scaler desde {scaler_path}. Error: {e}. Se procederá con datos no escalados.")
                scaler = None
        else:
            print("Advertencia: No se encontró el archivo del scaler. Se asumirá que los datos no necesitan ser escalados o ya lo están.")


        # 3. Añadir dimensión de canal para ConvLSTM
        # Los datos para ConvLSTM deben ser (muestras, pasos_tiempo, Alto, Ancho, canales)
        # Nuestras secuencias de entrada X son (muestras, sequence_length, Y, X, 1)
        # Por lo tanto, los datos base de donde se extraen las secuencias deben ser (T, Y, X, 1)
        precipitation_data_with_channel = np.expand_dims(precipitation_data_processed, axis=-1)
        # Forma: (num_total_data_steps, Y, X, 1)

        # --- Preparar la secuencia inicial para la predicción ---
        # Necesitamos los últimos SEQUENCE_LENGTH fotogramas de los datos históricos
        if num_total_data_steps < SEQUENCE_LENGTH:
            raise ValueError(f"No hay suficientes datos ({num_total_data_steps} pasos) para formar una secuencia inicial de longitud {SEQUENCE_LENGTH}.")

        # La secuencia más reciente del dataset
        # Esta es la secuencia que alimentará al modelo para la primera predicción futura
        current_sequence = precipitation_data_with_channel[num_total_data_steps - SEQUENCE_LENGTH : num_total_data_steps]
        # Forma: (SEQUENCE_LENGTH, Y, X, 1)

        # Cargar el modelo entrenado
        print(f"Cargando modelo entrenado desde: {MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Modelo cargado exitosamente.")

        # --- Bucle de Predicción Iterativa ---
        all_future_predictions_scaled_with_channel = [] # Para almacenar las predicciones (Y, X, 1)
        print(f"Iniciando predicción iterativa para {num_steps_to_predict} pasos...")

        temp_sequence = current_sequence.copy() # Trabajar con una copia para no alterar current_sequence original

        for i in range(num_steps_to_predict):
            # Reformar temp_sequence para la entrada del modelo: (1, SEQUENCE_LENGTH, Y, X, 1)
            input_for_model = np.expand_dims(temp_sequence, axis=0)

            # Predecir el siguiente fotograma
            # La salida del modelo es (1, Y, X, 1) para una predicción de un solo fotograma
            predicted_frame_batch_channel = model.predict(input_for_model)

            # Extraer el fotograma predicho (Y, X, 1)
            predicted_frame_channel = predicted_frame_batch_channel[0]
            
            # Almacenar la predicción (aún escalada, forma (Y, X, 1))
            # Esta es la predicción para last_data_date + (i+1) meses
            all_future_predictions_scaled_with_channel.append(predicted_frame_channel)

            # Actualizar temp_sequence: eliminar el fotograma más antiguo, añadir el nuevo fotograma predicho
            # predicted_frame_channel ya tiene la forma (Y, X, 1).
            # Necesitamos añadirle una dimensión de tiempo (como si fuera un fotograma) para concatenar.
            new_frame_for_sequence = np.expand_dims(predicted_frame_channel, axis=0) # Forma (1, Y, X, 1)
            temp_sequence = np.concatenate((temp_sequence[1:], new_frame_for_sequence), axis=0)
            # temp_sequence mantiene la forma (SEQUENCE_LENGTH, Y, X, 1)

            current_pred_month_date = last_data_date + relativedelta(months=i + 1)
            print(f"   Predicho para: {current_pred_month_date.strftime('%b %Y')}")

        # --- Post-procesar la predicción final objetivo ---
        # La predicción deseada es la última en all_future_predictions_scaled_with_channel
        final_prediction_scaled_with_channel = all_future_predictions_scaled_with_channel[-1] # Forma (Y, X, 1)
        final_prediction_scaled = final_prediction_scaled_with_channel[:, :, 0] # Forma (Y, X)

        # Invertir el escalado si se usó un scaler
        final_prediction_unscaled = final_prediction_scaled # Asumir no escalado por defecto
        if scaler is not None:
            print("Invirtiendo escalado de la predicción final.")
            # El scaler espera (n_samples, n_features) = (Y*X, 1) para inverse_transform
            # Reformar para la transformación inversa
            final_prediction_unscaled = scaler.inverse_transform(final_prediction_scaled.reshape(-1, 1)).reshape((y_dim_size, x_dim_size))
        
        # Asegurar que los valores predichos no sean negativos (la precipitación debe ser >= 0)
        final_prediction_unscaled[final_prediction_unscaled < 0] = 0
        print(f"Predicción completa para {target_pred_date.strftime('%b %Y')}.")

        # Visualizar la predicción final
        plt.figure(figsize=(10, 8)) # Ajusta el tamaño si es necesario
        
        plt.imshow(final_prediction_unscaled, cmap='Blues', origin='upper') # 'upper' para que el origen (0,0) del array esté arriba a la izquierda
        
        plt.title(f"Precipitación Predicha para {target_pred_date.strftime('%b %Y')}")
        plt.colorbar(label='Precipitación (mm/mes u otra unidad)') # Especifica la unidad
        
        # Configurar los ticks con las coordenadas geográficas formateadas
        # Determinando los ticks: Elegir algunos ticks uniformemente espaciados
        # Para el eje Y (Latitud)
        y_tick_indices = np.linspace(0, y_dim_size - 1, 5).astype(int) # 5 ticks (ajustable)
        y_tick_values = lat_coords[y_tick_indices]
        y_tick_labels = [lat_formatter(val, None) for val in y_tick_values]

        # Para el eje X (Longitud)
        x_tick_indices = np.linspace(0, x_dim_size - 1, 5).astype(int) # 5 ticks (ajustable)
        x_tick_values = lon_coords[x_tick_indices]
        x_tick_labels = [lon_formatter(val, None) for val in x_tick_values]

        plt.yticks(y_tick_indices, y_tick_labels)
        plt.xticks(x_tick_indices, x_tick_labels)

        plt.xlabel('Longitud')
        plt.ylabel('Latitud')

        plt.show()

        # Opcional: Guardar el array de datos predichos
        # prediction_savename = f"datos_prediccion_{target_pred_date.strftime('%Y_%m')}.npy"
        # np.save(prediction_savename, final_prediction_unscaled)
        # print(f"Array de datos predichos guardado como {prediction_savename}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en {DATA_PATH} o el archivo del modelo en {MODEL_SAVE_PATH}.")
    except ValueError as ve:
        print(f"Error de valor: {ve}")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la predicción: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Asegurarse de cerrar el dataset si se abrió
        if ds is not None:
            ds.close()


if __name__ == "__main__":
    # Asegúrate de que PROCESSED_DATA_DIR exista si el scaler está allí.
    # joblib.load manejará FileNotFoundError para el scaler_path en sí.
    # if not os.path.exists(PROCESSED_DATA_DIR):
    # print(f"Advertencia: El directorio de datos procesados '{PROCESSED_DATA_DIR}' podría ser necesario para el scaler.")
    main()
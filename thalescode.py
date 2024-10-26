import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
import unicodedata

def normalize_text(s):
    """
    Normaliza el texto eliminando acentos y convirtiendo a mayúsculas.
    """
    if isinstance(s, str):
        s = s.upper()
        #Normalizamos y eliminamos acentos para evitar errores y posibles fallas
        s = ''.join(
            c for c in unicodedata.normalize('NFKD', s)
            if not unicodedata.combining(c)
        )
        #Reemplazamos caracteres especiales específicos
        s = s.replace('Ñ', 'N')
        s = s.replace('Ç', 'C')
        s = s.replace('Ø', 'O')
        return s
    else:
        return s


#Especificamos la ruta del archivo CSV
ruta_archivo = r'C:\Users\Ángel\OneDrive\Desktop\Tercer Semestre\HackMX\carpetasFGJ_acumulado_2024_09.csv'

#Cargamos los datos desde el archivo CSV
df = pd.read_csv(ruta_archivo, low_memory=False)

#Seleccionamos las columnas necesarias
df = df[['fecha_hecho', 'alcaldia_hecho', 'delito']]

#Eliminamos filas con valores nulos
df = df.dropna(subset=['fecha_hecho', 'alcaldia_hecho', 'delito'])

#Convertimos 'fecha_hecho' a formato datetime
df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')

#Normalizamos las columnas de texto
df['alcaldia_hecho'] = df['alcaldia_hecho'].apply(normalize_text)
df['delito'] = df['delito'].apply(normalize_text)

#Obtenemos las listas únicas
delitos = df['delito'].unique()
alcaldias = df['alcaldia_hecho'].unique()

lista_predicciones = []

for delito in delitos:
    for alcaldia in alcaldias:
        #Filtramos los datos para el delito y alcaldía específicos
        df_filtrado = df[
            (df['delito'] == delito) &
            (df['alcaldia_hecho'] == alcaldia)
        ]

        #Agrupamos por fecha y contar el número de delitos
        df_agrupado = df_filtrado.groupby('fecha_hecho').size().reset_index(name='conteo_delitos')

        #Establecemos 'fecha_hecho' como índice y ordenar
        df_agrupado.set_index('fecha_hecho', inplace=True)
        df_agrupado = df_agrupado.sort_index()

        #Rellenamos fechas faltantes xon 0
        df_agrupado = df_agrupado.asfreq('D', fill_value=0)

        #Extraems la serie temporal
        serie = df_agrupado['conteo_delitos']

        #Verificamos si tenemos suficientes datos
        if len(serie) >= 2:
            try:
                #Aplicamos el modelo de Suavización Exponencial Simple
                modelo = SimpleExpSmoothing(serie).fit(smoothing_level=0.5, optimized=False)

                #Predecimos el siguiente día
                prediccion = modelo.forecast(1)

                #Preparamos el DataFrame de predicción
                df_pred = prediccion.reset_index()
                df_pred.columns = ['fecha_hecho', 'prediccion_delitos']
                df_pred['alcaldia_hecho'] = alcaldia
                df_pred['delito'] = delito

                #Añadimos la predicción a la lista
                lista_predicciones.append(df_pred)
                print(f"Predicción generada para {delito} en {alcaldia}")
            except Exception as e:
                print(f"Error al predecir para {delito} en {alcaldia}: {e}")
        else:
            print(f"No hay suficientes datos para {delito} en {alcaldia}")

if lista_predicciones:
    df_predicciones_total = pd.concat(lista_predicciones, ignore_index=True)
else:
    print("No se generaron predicciones.")

#Guardamos las predicciones en un archivo CSV
df_predicciones_total.to_csv('predicciones_delitos_todos.csv', index=False)
print("Predicciones guardadas en 'predicciones_delitos_todos.csv'")
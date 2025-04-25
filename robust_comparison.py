import cv2
import mediapipe as mp
import numpy as np
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.model_selection import KFold 
from util import extraer_caracteristicas
from compare1 import procesar_video,  cargar_datos, convertir_a_vector, calcular_distancia_dtw

def calcular_umbral(archivos_video, etiquetas, nombres_clases, k_folds=3):
    """
    Realiza K-Fold Cross-Validation para calcular un umbral óptimo.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)  # shuffle=True para mezclar los datos

    distancias_intra_clase_todas = []
    distancias_inter_clase_todas = []
    umbral_optimo = None
    precision_maxima = 0.0

    for fold, (train_index, test_index) in enumerate(kf.split(archivos_video)):
        print(f"Fold: {fold + 1}")

        archivos_video_train = [archivos_video[i] for i in train_index]
        etiquetas_train = [etiquetas[i] for i in train_index]
        archivos_video_test = [archivos_video[i] for i in test_index]
        etiquetas_test = [etiquetas[i] for i in test_index]

        # 1. Distancias Intra-Clase (Dentro de la Misma Clase)
        distancias_intra_clase = []
        for i in range(len(archivos_video_train)):
            for j in range(i + 1, len(archivos_video_train)):
                print("folder: ", fold+1, "i: ", i, "j: ", j, "len_archivos: ", len(archivos_video_train))
                if etiquetas_train[i] == etiquetas_train[j]:
                    vector1 = procesar_video(archivos_video_train[i])
                    vector2 = procesar_video(archivos_video_train[j])

                    if vector1 is None or vector2 is None:
                        continue
                    # Elimina diccionarios vacíos (frames sin detección o errores) y convierte a vectores
                    vectores_video1 = [convertir_a_vector(c) for c in vector1 if c]
                    vectores_video2 = [convertir_a_vector(c) for c in vector2 if c]

                    # Elimina los vectores None (fallos en la extracción de características)
                    vectores_video1 = [v for v in vectores_video1 if v is not None]
                    vectores_video2 = [v for v in vectores_video2 if v is not None]

                    # Verifica si hay suficientes frames para comparar
                    if len(vectores_video1) < 2 or len(vectores_video2) < 2:
                        print("No hay suficientes frames con detección de pose en uno o ambos videos.  No se puede comparar.")
                        continue

                    distancia = calcular_distancia_dtw(vectores_video1, vectores_video2)
                    print(distancia)
                    distancias_intra_clase.append(distancia)
        distancias_intra_clase_todas.extend(distancias_intra_clase)


        # 2. Distancias Inter-Clase (Entre Diferentes Clases)
        distancias_inter_clase = []
        for i in range(len(archivos_video_train)):
            for j in range(i + 1, len(archivos_video_train)):
                if etiquetas_train[i] != etiquetas_train[j]:
                    vector1 = procesar_video(archivos_video_train[i])
                    vector2 = procesar_video(archivos_video_train[j])
                    if vector1 is None or vector2 is None:
                        continue
                    # Elimina diccionarios vacíos (frames sin detección o errores) y convierte a vectores
                    vectores_video1 = [convertir_a_vector(c) for c in vector1 if c]
                    vectores_video2 = [convertir_a_vector(c) for c in vector2 if c]

                    # Elimina los vectores None (fallos en la extracción de características)
                    vectores_video1 = [v for v in vectores_video1 if v is not None]
                    vectores_video2 = [v for v in vectores_video2 if v is not None]

                    # Verifica si hay suficientes frames para comparar
                    if len(vectores_video1) < 2 or len(vectores_video2) < 2:
                        print("No hay suficientes frames con detección de pose en uno o ambos videos.  No se puede comparar.")
                        continue

                    distancia = calcular_distancia_dtw(vectores_video1, vectores_video2)
                    distancias_inter_clase.append(distancia)
        distancias_inter_clase_todas.extend(distancias_inter_clase)

        # 3. Calcular el Umbral (Usando las Distancias Intra-Clase y Inter-Clase)
        #   Aquí, se busca un umbral que maximize la separación entre las distancias intra-clase y inter-clase.
        #   Este es un enfoque simple; podrías usar métodos más sofisticados si es necesario.

        # Generar una lista de posibles umbrales (entre el mínimo y el máximo de las distancias)
        umbrales_posibles = np.linspace(min(distancias_intra_clase + distancias_inter_clase),
                                        max(distancias_intra_clase + distancias_inter_clase),
                                        100) # 100 umbrales diferentes

        # Evaluar cada umbral en el conjunto de prueba
        precisiones = []
        for umbral in umbrales_posibles:
            # Predecir las etiquetas del conjunto de prueba
            predicciones = []
            for i in range(len(archivos_video_test)):
                for j in range(i + 1, len(archivos_video_test)):
                    vector1 = procesar_video(archivos_video_test[i])
                    vector2 = procesar_video(archivos_video_test[j])
                    if vector1 is None or vector2 is None:
                        continue
                    # Elimina diccionarios vacíos (frames sin detección o errores) y convierte a vectores
                    vectores_video1 = [convertir_a_vector(c) for c in vector1 if c]
                    vectores_video2 = [convertir_a_vector(c) for c in vector2 if c]

                    # Elimina los vectores None (fallos en la extracción de características)
                    vectores_video1 = [v for v in vectores_video1 if v is not None]
                    vectores_video2 = [v for v in vectores_video2 if v is not None]

                    # Verifica si hay suficientes frames para comparar
                    if len(vectores_video1) < 2 or len(vectores_video2) < 2:
                        print("No hay suficientes frames con detección de pose en uno o ambos videos.  No se puede comparar.")
                        continue

                    distancia = calcular_distancia_dtw(vectores_video1, vectores_video2)
                    if distancia <= umbral:
                        # Si la distancia es menor o igual al umbral, predecir que son de la misma clase
                        prediccion = (etiquetas_test[i] == etiquetas_test[j]) # True si son la misma clase, False si no
                    else:
                        # Si la distancia es mayor al umbral, predecir que son de diferentes clases
                        prediccion = (etiquetas_test[i] != etiquetas_test[j]) # True si NO son la misma clase, False si lo son
                    predicciones.append(prediccion)


            # Calcular la precisión
            precision = np.mean(predicciones) #Porcentaje de predicciones correctas
            precisiones.append(precision)

        # Seleccionar el umbral que da la máxima precisión en este fold
        indice_mejor_umbral = np.argmax(precisiones)
        umbral_fold = umbrales_posibles[indice_mejor_umbral]
        precision_fold = precisiones[indice_mejor_umbral]

        print(f"  Mejor umbral para este fold: {umbral_fold}, Precisión: {precision_fold}")

        # Guardar el mejor umbral si es el mejor hasta ahora
        if precision_fold > precision_maxima:
            precision_maxima = precision_fold
            umbral_optimo = umbral_fold

    print("\nResultados de la Validación Cruzada:")
    print(f"  Umbral Óptimo (promedio de todos los folds): {umbral_optimo}")

    # Análisis de las distancias intra e inter clase (después de K-fold)
    distancias_intra_clase_todas = np.array(distancias_intra_clase_todas)
    distancias_inter_clase_todas = np.array(distancias_inter_clase_todas)

    print("\nAnálisis de Distancias Intra-Clase:")
    print(f"  Media: {np.mean(distancias_intra_clase_todas)}")
    print(f"  Desviación Estándar: {np.std(distancias_intra_clase_todas)}")

    print("\nAnálisis de Distancias Inter-Clase:")
    print(f"  Media: {np.mean(distancias_inter_clase_todas)}")
    print(f"  Desviación Estándar: {np.std(distancias_inter_clase_todas)}")


    return umbral_optimo

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Ruta al directorio principal de tus datos
directorio_datos = 'ixmas'

# Cargar los datos
archivos_video, etiquetas, nombres_clases = cargar_datos(directorio_datos)

# Calcular el umbral usando K-Fold Cross-Validation
umbral = calcular_umbral(archivos_video, etiquetas, nombres_clases, k_folds=5)

print(f"\nUmbral Final Sugerido (basado en K-Fold): {umbral}")

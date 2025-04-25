import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import glob

from util import extraer_caracteristicas


def procesar_video(video_path):
    """Procesa un video y extrae las características de la pose."""
    caracteristicas_video = []
    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                image_height, image_width, _ = image.shape
                landmarks = results.pose_landmarks.landmark
                caracteristicas = extraer_caracteristicas(landmarks, image_width, image_height)
                if caracteristicas:  # Asegúrate de que las características no sean None
                    caracteristicas_video.append(caracteristicas)

                # Opcional: Dibuja la pose y las características en la imagen para visualización
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # cv2.putText(...)  # Agrega texto con los ángulos y distancias
                cv2.imshow('Video', image)
            else:
                # Si no se detecta la pose, agrega un diccionario vacío o un vector cero.
                # Esto depende de cómo quieras manejar los frames sin detección.
                caracteristicas_video.append({})  # O np.zeros(N) si usas vectores de longitud fija

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    return caracteristicas_video

def convertir_a_vector(caracteristicas):
    """Convierte un diccionario de características a un vector numpy."""
    if not caracteristicas:
        return None # Maneja el caso de diccionarios vacíos
    return np.array(list(caracteristicas.values()))

def calcular_distancia_dtw(vector1, vector2):
    """Calcula la distancia DTW entre dos secuencias de vectores."""
    _, path = fastdtw(vector1, vector2, dist=euclidean)
    
    distancias_frame_a_frame = [euclidean(vector1[i], vector2[j]) for i, j in path]
    distancia_promedio = np.mean(distancias_frame_a_frame)

    return distancia_promedio

def cargar_datos(directorio_datos, extension = '*.avi'):
    """
    Carga los datos desde el directorio, creando listas de archivos de video
    y sus etiquetas de clase.
    """
    archivos_video = []
    etiquetas = []
    nombres_clases = []

    for i, nombre_carpeta in enumerate(os.listdir(directorio_datos)):
        ruta_carpeta = os.path.join(directorio_datos, nombre_carpeta)

        if not os.path.isdir(ruta_carpeta):
            continue
        
        # Guardar el nombre de la clase
        nombres_clases.append(nombre_carpeta)  
        
        archivos_clase = glob.glob(os.path.join(ruta_carpeta, extension))
        archivos_video.extend(archivos_clase)

        # Etiqueta con el índice de la clase
        etiquetas.extend([i] * len(archivos_clase))  

    return archivos_video, etiquetas, nombres_clases

def convertir_a_vector(caracteristicas):
    """Convierte un diccionario de características a un vector numpy."""
    if not caracteristicas:
        return None # Maneja el caso de diccionarios vacíos
    return np.array(list(caracteristicas.values()))

def calcular_umbral(archivos_video, etiquetas, nombres_clases, k_folds=3):
    """
    Calcula un umbral óptimo.
    """
    
    distancias_intra_clase_todas = []

    distancias_intra_clase = []
    for i in range(len(archivos_video)):
        for j in range(i + 1, len(archivos_video)):
            print("i: ", i, "j: ", j, "len_archivos: ", len(archivos_video))
            if etiquetas[i] == etiquetas[j]:
                vector1 = procesar_video(archivos_video[i])
                vector2 = procesar_video(archivos_video[j])

                if vector1 is None or vector2 is None:
                    continue

                # Elimina diccionarios vacíos (frames sin detección o errores) y convierte a vectores los restantes
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
                print("distancia: ", distancia)
                distancias_intra_clase.append(distancia)

    distancias_intra_clase_todas.extend(distancias_intra_clase)
    print("\nAnálisis de Distancias Intra-Clase:")

    distancias_intra_clase = np.array(distancias_intra_clase) 

    # Calcula estadísticas de las distancias intra-clase
    media = np.mean(distancias_intra_clase)
    desviacion_estandar = np.std(distancias_intra_clase)
    percentil_95 = np.percentile(distancias_intra_clase, 95)

    print(f"  Media: {media}")
    print(f"  Desviación Estándar: {desviacion_estandar}")
    print(f"  Percentil 95: {percentil_95}")

    # Define el umbral (ajusta según tus necesidades)
    umbral = media + 2 * desviacion_estandar  # Un ejemplo
    
    return umbral


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Ruta al directorio principal de los datos
directorio_datos = 'ixmas'

# Carga los datos
#archivos_video, etiquetas, nombres_clases = cargar_datos(directorio_datos)

# Calcula el umbral 
#umbral = calcular_umbral(archivos_video, etiquetas, nombres_clases, k_folds=5)

#print(f"\nUmbral Final Sugerido (basado en K-Fold): {umbral}")

    
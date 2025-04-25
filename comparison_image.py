from weak_comparison import extraer_caracteristicas, cargar_datos, convertir_a_vector
from scipy.spatial.distance import euclidean

import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def dist(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        a = vec1[i] - vec2[i]
        sum = sum + a**2
    return math.sqrt(sum)

def procesar_image(image_path):
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        landmarks = results.pose_landmarks.landmark
        caracteristicas = extraer_caracteristicas(landmarks, width, height)
    
    return convertir_a_vector(caracteristicas)

def calcular_umbral_images(archivos_image, etiquetas):
    distancias_intra_clase_todas = []
    umbrales = []
    diferencia_caracteristicas = []

    for i in range(len(archivos_image)):
        for j in range(i + 1, len(archivos_image)):
            print("i: ", i, "j: ", j, "len_archivos: ", len(archivos_image))

            if etiquetas[i] == etiquetas[j]:
                vector1 = procesar_image(archivos_image[i])
                vector2 = procesar_image(archivos_image[j])

                if vector1 is None or vector2 is None:
                    continue

                distancias_intra_clase_todas.append(dist(vector1, vector2))
                diferencia_caracteristicas.append(abs(vector1 - vector2))
                print('distancia: ', dist(vector1, vector2))

    distancias = np.array(distancias_intra_clase_todas)
    media = np.mean(distancias)
    desviacion_estandar = np.std(distancias)
    results = []
    print("media: ", media, "desviacion: ", desviacion_estandar)

    if len(diferencia_caracteristicas) > 0:
        for i in range(len(diferencia_caracteristicas[0])):
            umbrales = []
            for j in range(len(diferencia_caracteristicas)):
                umbrales.append(diferencia_caracteristicas[j][i])
            
            umbrales = np.array(umbrales)
            value = np.mean(umbrales)
            results.append(value)
            
    print(results)
    
archivos_image, etiquetas, nombres_clases = cargar_datos("bd", "*.jpg")

calcular_umbral_images(archivos_image, etiquetas)


import mediapipe as mp
import numpy as np
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Funciones Auxiliares ---

def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (en grados)."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    angle = angle % 360
    if angle > 180:
        angle = 360 - angle
    return angle

def calcular_distancia(a, b):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def extraer_caracteristicas(landmarks, image_width, image_height):
    """Extrae ángulos y distancias normalizadas de los landmarks."""
    caracteristicas = {}

    # Obtiene las coordenadas de los puntos clave
    try:  # Manejo de errores para puntos clave faltantes
        hombro_izquierdo = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height]
        hombro_derecho = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height]
        codo_izquierdo = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height]
        codo_derecho = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height]
        muneca_izquierda = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height]
        muneca_derecha = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height]
        cadera_izquierda = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height]
        cadera_derecha = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height]
        rodilla_izquierda = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height]
        rodilla_derecha = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height]
        tobillo_izquierdo = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height]
        tobillo_derecho = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height]
    except:
        return None  # Devuelve None si algún punto clave falta

    # Calcula los ángulos
    caracteristicas['angulo_codo_izquierdo'] = calcular_angulo(hombro_izquierdo, codo_izquierdo, muneca_izquierda)
    caracteristicas['angulo_codo_derecho'] = calcular_angulo(hombro_derecho, codo_derecho, muneca_derecha)
    caracteristicas['angulo_rodilla_izquierda'] = calcular_angulo(cadera_izquierda, rodilla_izquierda, tobillo_izquierdo)
    caracteristicas['angulo_rodilla_derecha'] = calcular_angulo(cadera_derecha, rodilla_derecha, tobillo_derecho)

    # Calcula la distancia de referencia (distancia entre los hombros)
    distancia_hombros = calcular_distancia(hombro_izquierdo, hombro_derecho)
    distancia_munecas = calcular_distancia(muneca_izquierda, muneca_derecha)
    distancia_tobillos = calcular_distancia(tobillo_izquierdo, tobillo_derecho)

    # Calcula las distancias normalizadas
    caracteristicas['distancia_cadera_rodilla_izquierda'] = calcular_distancia(cadera_izquierda, rodilla_izquierda) / (distancia_hombros + 1e-6)  # Evita la división por cero
    caracteristicas['distancia_cadera_rodilla_derecha'] = calcular_distancia(cadera_derecha, rodilla_derecha) / (distancia_hombros + 1e-6)
    caracteristicas['distancia_munecas'] = distancia_munecas/ (distancia_hombros + 1e-6)
    caracteristicas['distancia_tobillos'] = distancia_tobillos/ (distancia_hombros + 1e-6)
    
    caracteristicas['angulo_hombro_izquierdo'] = calcular_angulo(codo_izquierdo, hombro_izquierdo, cadera_izquierda)
    caracteristicas['angulo_hombro_derecho'] = calcular_angulo(codo_derecho, hombro_derecho, cadera_derecha)
    #caracteristicas['pendiente_muneca_codo_izquierdo'] = calcular_pendiente(muneca_izquierda, codo_izquierdo)
    #caracteristicas['pendiente_muneca_codo_derecho'] = calcular_pendiente(muneca_derecha, codo_derecho)
    #caracteristicas['pendiente_codo_hombro_izquierdo'] = calcular_pendiente(codo_izquierdo, hombro_izquierdo)
    #caracteristicas['pendiente_codo_hombro_derecho'] = calcular_pendiente(codo_derecho, hombro_derecho)
    #caracteristicas['pendiente_cadera_rodilla_izquierda'] = calcular_pendiente(cadera_izquierda, rodilla_izquierda)
    #caracteristicas['pendiente_cadera_rodilla_derecha'] = calcular_pendiente(cadera_derecha, rodilla_derecha)
    #caracteristicas['pendiente_rodilla_tobillo_izquierdo'] = calcular_pendiente(rodilla_izquierda, tobillo_izquierdo)
    #caracteristicas['pendiente_rodilla_tobillo_derecho'] = calcular_pendiente(rodilla_derecha, tobillo_derecho)

    return caracteristicas

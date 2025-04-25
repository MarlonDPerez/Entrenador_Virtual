
import cv2
import mediapipe as mp 
import numpy as np
from scipy.spatial.distance import euclidean
from weak_comparison import convertir_a_vector
from util import calcular_angulo, calcular_distancia, extraer_caracteristicas
from mistral import descripcion_vector, resumen
from answers import predefinded_answers

def detectar_postura(frame):
    """Detecta los puntos clave de la postura en un frame usando MediaPipe (o OpenPose).
       Devuelve una lista de coordenadas (x, y) de los puntos clave.
       Si no se detecta postura, devuelve None.
    """
    global pose  # Asegúrate de que 'pose' esté inicializado globalmente
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        puntos_clave = [(lm.x, lm.y) for lm in landmarks]  # Normalizadas (0 a 1)
        image = frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        carac =extraer_caracteristicas(landmarks, image_width, image_height)
        return puntos_clave, carac
    else: 
        return None, None

def normalizar_postura(puntos_clave):
    """Normaliza los puntos clave para que sean independientes del tamaño y posición.
    """
    hombro_izquierdo = puntos_clave[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hombro_derecho = puntos_clave[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    punto_medio = ((hombro_izquierdo[0] + hombro_derecho[0]) / 2,
                   (hombro_izquierdo[1] + hombro_derecho[1]) / 2)

    puntos_normalizados = [(x - punto_medio[0], y - punto_medio[1]) for x, y in puntos_clave]

    return puntos_normalizados

def comparar_posturas(postura_referencia, postura_actual):
    """Compara dos posturas (listas de puntos clave normalizados).
       Devuelve una puntuación de similitud (menor = más similar).
    """
    if postura_referencia is None or postura_actual is None:
        return float('inf')

    distancias = [euclidean(ref, actual) for ref, actual in zip(postura_referencia, postura_actual)]
    return np.mean(distancias)

# --- Variables Globales y Parámetros ---
SIMILARITY_THRESHOLD = 0.04

POSE_VARIATION_THRESHOLD = 0.007 #Umbral para detectar variación significativa en la postura

def analizar_video_referencia(video_path, max_frames=1000):
    """
        Analiza el video de referencia y extrae poses representativas.
    """
    cap = cv2.VideoCapture(video_path)
    frames_representativos = []
    carac_representativas = []
    postura_anterior = None
    carac_anterior = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        puntos_clave, carac = detectar_postura(frame)

        if puntos_clave:
            postura_normalizada = normalizar_postura(puntos_clave)
            carac_ = convertir_a_vector(carac)

            if postura_anterior is None:
                #Primer frame
                frames_representativos.append(postura_normalizada)
                postura_anterior = postura_normalizada
                carac_representativas.append(carac_)
                carac_anterior = carac_
            else:
                distancia = comparar_posturas(postura_anterior, postura_normalizada)
                distancia1 = euclidean(carac_anterior, carac_)

                if distancia > POSE_VARIATION_THRESHOLD:
                #if distancia1 > 90:
                    #Postura significativamente diferente, guardar
                    frames_representativos.append(postura_normalizada)
                    postura_anterior = postura_normalizada
                    carac_representativas.append(carac_)
                    carac_anterior - carac_

        frame_count += 1
    cap.release()
    return frames_representativos, carac_representativas

# --- 4. Main Modificado ---

def main():

    mp_drawing = mp.solutions.drawing_utils
    global mp_pose
    mp_pose = mp.solutions.pose
    global pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Inicializa 'pose' aquí

    # Analizar el video de referencia
    video_referencia_path = "video_cortado.mp4"
    frames_referencia, carac_referencia = analizar_video_referencia(video_referencia_path)
    print(len(carac_referencia))

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    frame_referencia_actual = 0
    en_movimiento = False
    postura_anterior = None #Para detectar cuando la postura del usuario cambia
    feedback_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar postura en el frame actual de la cámara
        puntos_clave_actual, carac1 = detectar_postura(frame)

        if puntos_clave_actual:
            puntos_normalizados_actual = normalizar_postura(puntos_clave_actual)
            carac1_ = convertir_a_vector(carac1)
            
            # Si no estamos en movimiento, buscar el inicio del movimiento
            if not en_movimiento:
                # Comparar con el primer frame de referencia
                distancia = comparar_posturas(frames_referencia[0], puntos_normalizados_actual)
                distancia1 = euclidean(carac_referencia[0], carac1_)
                
                if distancia < SIMILARITY_THRESHOLD:
                    en_movimiento = True
                    frame_referencia_actual = 0
                    print("Movimiento detectado! Empezando la comparación.")
                    postura_anterior = puntos_normalizados_actual #Guardamos la primera postura detectada

            # Si estamos en movimiento, comparar con el frame correspondiente
            if en_movimiento:
                #Comparar si la pose actual ha cambiado significativamente desde la ultima comparacion.
                if postura_anterior is None or comparar_posturas(postura_anterior, puntos_normalizados_actual) > POSE_VARIATION_THRESHOLD:
                    #distancia = comparar_posturas(frames_referencia[frame_referencia_actual], puntos_normalizados_actual)
                    distancia1 = euclidean(carac_referencia[frame_referencia_actual], carac1_)
                    
                    #if distancia > RESET_THRESHOLD:
                    if distancia1 > 66:
                        print("Postura muy diferente!...")

                    vec = carac1_ - carac_referencia[frame_referencia_actual]
                        
                    print(f"Frame Representativo: {frame_referencia_actual}, Similitud: {distancia1}")
                    
                    feedback = predefinded_answers(vec)
                    print('feedback: \n', feedback)
                    
                    feedback = 'frame número ' + str(frame_referencia_actual) + ':\n' + feedback
                    feedback_frames.append(feedback)
                    frame_referencia_actual += 1
                    #Actualizar la postura anterior
                    postura_anterior = puntos_normalizados_actual 
                    
                    # Si llegamos al final del video de referencia, volver a empezar
                    if frame_referencia_actual >= len(frames_referencia):
                        print("Video de referencia completado! Reiniciando...")
                        cv2.imshow('Video de referencia completado! Reiniciando...', frame)

                        frame_referencia_actual = 0
                        en_movimiento = False  # Opcional: volver a buscar el inicio
                        print(resumen(feedback_frames))
                    
                        feedback_frames = []
                        postura_anterior = None

        # Visualización (opcional)
        cv2.imshow('Corrección de Postura en Tiempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()

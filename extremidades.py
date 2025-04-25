import cv2
import mediapipe as mp 
import numpy as np

global mp_pose
mp_pose = mp.solutions.pose
global pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

extremidades = {
    'brazo_superior_izquierdo': ('hombro_izquierdo', 'codo_izquierdo'),
    'brazo_superior_derecho': ('hombro_derecho', 'codo_derecho'),
    'brazo_inferior_izquierdo': ('codo_izquierdo', 'muneca_izquierda'),
    'brazo_inferior_derecho': ('codo_derecho', 'muneca_derecha'),
    'muslo_izquierdo': ('cadera_izquierda', 'rodilla_izquierda'),
    'muslo_derecho': ('cadera_derecha', 'rodilla_derecha'),
    'pierna_izquierda': ('rodilla_izquierda', 'tobillo_izquierdo'),
    'pierna_derecha': ('rodilla_derecha', 'tobillo_derecho'),
    'torso_izquierdo': ('hombro_izquierdo', 'cadera_izquierda'),
    'torso_derecho': ('hombro_derecho', 'cadera_derecha')
}

angulos = {
    'brazo_superior_izquierdo': ('cadera_izquierda', 'hombro_izquierdo', 'codo_izquierdo'),
    'brazo_superior_derecho': ('cadera_derecha', 'hombro_derecho', 'codo_derecho'),
    'brazo_inferior_izquierdo': ('hombo_izquierdo', 'codo_izquierdo', 'muneca_izquierda'),
    'brazo_inferior_derecho': ('hombo_derecho', 'codo_derecho', 'muneca_derecha'),
    'muslo_izquierdo': ('cadera_derecha', 'cadera_izquierda', 'rodilla_izquierda'),
    'muslo_derecho': ('cadera_izquierda', 'cadera_derecha', 'rodilla_derecha'),
    'pierna_izquierda': ('cadera_izquierda', 'rodilla_izquierda', 'tobillo_izquierdo'),
    'pierna_derecha': ('cadera_derecha', 'rodilla_derecha', 'tobillo_derecho'),
    'torso_izquierdo': ('hombro_izquierdo', 'cadera_izquierda'),
    'torso_derecho': ('hombro_derecho', 'cadera_derecha')
}
def articulaciones(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, image = cap.read()
        if not ret:
            break
            
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
                dic_ = {}
                image_height, image_width, _ = image.shape
                landmarks = results.pose_landmarks.landmark
                dic_['hombro_izquierdo'] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height]
                dic_['hombro_derecho'] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height]
                dic_['codo_izquierdo'] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height]
                dic_['codo_derecho'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height]
                dic_['muneca_izquierda'] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height]
                dic_['muneca_derecha'] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height]
                dic_['cadera_izquierda'] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height]
                dic_['cadera_derecha'] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height]
                dic_['rodilla_izquierda'] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height]
                dic_['rodilla_derecha'] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height]
                dic_['tobillo_izquierdo'] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height]
                dic_['tobillo_derecho'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height]
                frames.append(dic_)
    
    return frames

def variacion(articulaciones_frames):
    variaciones = {}

    for extremidad, puntos in extremidades.items():
        punto_inicial_nombre, punto_final_nombre = puntos
        magnitudes = []

        for frame in articulaciones_frames:
            if punto_inicial_nombre in frame and punto_final_nombre in frame:
                x1, y1 = frame[punto_inicial_nombre]
                x2, y2 = frame[punto_final_nombre]

                angulo_rad = np.arctan2(y2 - y1, x2 - x1)
                # Convertir a grados
                angulo_deg = np.degrees(angulo_rad)
                magnitudes.append(angulo_deg)

        if magnitudes:
            variacion = np.max(magnitudes) - np.min(magnitudes)
            variaciones[extremidad] = variacion
        else:
            variaciones[extremidad] = 0
    return variaciones  

frames = articulaciones("video_cortado.mp4")
dic_ = variacion(frames)
print(dic_)
print('#############################################')
sorted_dicc = sorted(dic_.items(), key = lambda item: item[1], reverse=True)

top_5 = sorted_dicc[:5]

print(top_5)
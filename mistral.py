from mistralai import Mistral

def show_info(vector_descripcion, vector_valores):
    MISTRAL_API_KEY='amk4WhcFI6dXBHJUzcjhQhwUhmgjMqlP'
    client = Mistral(api_key=MISTRAL_API_KEY)

    instrucciones = (
        '''A continuación, te proporcionaré un vector que representa las diferencias en la postura de una persona en comparación con una postura de referencia. Cada componente del vector corresponde a una característica específica.  Debes generar una respuesta concisa y clara indicando las correcciones que la persona debe realizar en su postura, como lo haría un entrenador en el gymnasio'''
    )
    descripcion = f"Descripción del vector: {vector_descripcion}"
    valores = f"Valores del vector: {vector_valores}"

    prompt = f"{instrucciones}\n{descripcion}\n{valores}\nCorrecciones:"


    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al llamar a la API de Mistral: {e}")
        return "Lo siento, no puedo generar el comercial en este momento debido a un error."

    
descripcion_vector = [
    "Ángulo del codo izquierdo (grados)",
    "Ángulo del codo derecho (grados)",
    "Ángulo de la rodilla izquierda (grados)",
    "Ángulo de la rodilla derecha (grados)",
    "Distancia de la cadera a la rodilla izquierda",
    "Distancia de la cadera a la rodilla derecha"
    ]
def resumen(feedback_frames):
    MISTRAL_API_KEY='amk4WhcFI6dXBHJUzcjhQhwUhmgjMqlP'
    client = Mistral(api_key=MISTRAL_API_KEY)
    instruccion = 'A continuacion te proporcionaré información que representa sugerencias hechas a una persona que está haciendo un ejercicio en tiempo real y se está comparando con un video de referencia, en cada frame comparado te diré las sugerencias hechas. A modo general has un resumen de estas sugerencias explicando a la persona que debe corregir. Estas son las sugerencias por cada frame:\n'
    x = int(len(feedback_frames)/10)
    i = 0
    while i < len(feedback_frames):
        instruccion = instruccion + feedback_frames[i] + '\n'
        i = i + x
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": instruccion}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al llamar a la API de Mistral: {e}")
        return "Lo siento, no puedo generar el comercial en este momento debido a un error."
    
##################################################################
#                     TEST_FOR_TIME_OF_LLM                       #
##################################################################

import random
import time
import numpy as np

def test_vector():
    times = []
    for i in range(100):
        vector = []
        vector.append(random.uniform(-10, 10))
        vector.append(random.uniform(-3, 3))
        vector.append(random.uniform(-50, 50))
        vector.append(random.uniform(-55, 50))
        vector.append(random.uniform(-4, 4))
        vector.append(random.uniform(4, 4))
        inicio = time.time()
        show_info(descripcion_vector, vector)
        fin = time.time()
        duration = fin - inicio
        times.append(duration)
        print(duration)
    sum = 0
    for i in range(len(times)):
        sum = sum + times[i]
    print('media: ', sum/len(times))
#test_vector()
from possible_answers import *
import random

def predefinded_answers(differences_vector):
    answer = ""
    
    if differences_vector[0] > 12:
        answer = answer + random.choice(possible_answers_0_g) + "\n"
    
    if differences_vector[0] < -12:
        answer = answer + random.choice(possible_answers_0_l) + "\n"
    
    if differences_vector[1] > 12:
        answer = answer + random.choice(possible_answers_1_g) + "\n"
    
    if differences_vector[1] < -12:
        answer = answer + random.choice(possible_answers_1_l) + "\n"
    
    if differences_vector[2] > 23:
        answer = answer + random.choice(possible_answers_2_g) + "\n"
    
    if differences_vector[2] < -23:
        answer = answer + random.choice(possible_answers_2_l) + "\n"
    
    if differences_vector[3] > 23:
        answer = answer + random.choice(possible_answers_3_g) + "\n"
    
    if differences_vector[3] < -23:
        answer = answer + random.choice(possible_answers_3_l) + "\n"
    
    if differences_vector[4] > 5:
        answer = answer + random.choice(possible_answers_4_g) + "\n"
    
    if differences_vector[4] < -5:
        answer = answer + random.choice(possible_answers_4_l) + "\n"
    
    if differences_vector[5] > 6:
        answer = answer + random.choice(possible_answers_5_g) + "\n"
    
    if differences_vector[5] < -6:
        answer = answer + random.choice(possible_answers_5_l) + "\n"
    
    if differences_vector[6] > 2.4:
        answer = answer + random.choice(possible_answers_6_g) + "\n"
    
    if differences_vector[6] < -2.4:
        answer = answer + random.choice(possible_answers_6_l) + "\n"

    if differences_vector[7] > 9.9:
        answer = answer + random.choice(possible_answers_7_g) + "\n"

    if differences_vector[7] < -9.9:
        answer = answer + random.choice(possible_answers_7_l) + "\n"

    if differences_vector[8] > 20:
        answer = answer + random.choice(possible_answers_8_g) + "\n"
    
    if differences_vector[8] < -20:
        answer = answer + random.choice(possible_answers_8_l) + "\n"

    if differences_vector[9] > 16:
        answer = answer + random.choice(possible_answers_9_g) + "\n"
    if differences_vector[9] < -16:
        answer = answer + random.choice(possible_answers_9_l) + "\n"
    
    if answer == "":
        answer = "Bien hecho, pose sin errores."
    
    return answer


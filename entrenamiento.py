import cv2
import os
import numpy as np
import time


def obtener_modelo(method, faces_data, labels):

    if method == 'EigenFaces':
        emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces':
        emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH':
        emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenamiento
    print(f'Entrenando {method}'.center(50, '-'))
    inicio = time.time()
    emotion_recognizer.train(faces_data, np.array(labels))
    tiempo_entrenamiento = time.time() - inicio
    print(f'Tiempo de entrenamiento para {method}: {tiempo_entrenamiento}')

    # Almacenando modelo
    emotion_recognizer.write(f'modelo{method}.xml')


data_path = 'D:/UdeA_ITM/Inteligencia Artificial/Vision Artificial/' \
           'TrabajoFinal/FacialRecognition/Data'
emotions_list = os.listdir(data_path)
print(f'Lista de emociones {emotions_list}')

labels = []
faces_data = []
label = 0

for name_dir in emotions_list:
    emotions_path = data_path + '/' + name_dir

    for file_name in os.listdir(emotions_path):
        labels.append(label)
        faces_data.append(cv2.imread(emotions_path + '/' + file_name, 0))

    label += 1

obtener_modelo('EigenFaces', faces_data, labels)
obtener_modelo('FisherFaces', faces_data, labels)
obtener_modelo('LBPH', faces_data, labels)





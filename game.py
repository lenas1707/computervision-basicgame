import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import math

print("Iniciando o jogo...")
print("Carregando bibliotecas...")

draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score = 0

x_enemy = random.randint(50,600)
y_enemy = random.randint(50,400)

COR_ALVO = (100, 200, 100)
COR_DEDO = (200, 100, 100)
COR_PLACAR = (200, 200, 200)
COR_LANDMARKS = (150, 150, 150)

def enemy(image):
    global x_enemy, y_enemy
    cv2.circle(image, (x_enemy, y_enemy), 25, COR_ALVO, 5)

def check_collision(x1, y1, x2, y2, threshold=50):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance < threshold

print("Iniciando câmera...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erro: Não foi possível abrir a câmera!")
    exit()

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Iniciando detecção de mãos...")
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    print("Jogo iniciado! Pressione 'q' para sair.")
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame is None:
            print("Erro ao ler frame da câmera")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image_height, image_width, _ = image.shape
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(image, f"Score: {score}", (400,30), font, 1, COR_PLACAR, 2, cv2.LINE_AA)
        enemy(image)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                  draw.DrawingSpec(color=COR_LANDMARKS, thickness=2, circle_radius=2))

                index_finger = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pixelCoord = draw._normalized_to_pixel_coordinates(
                    index_finger.x, 
                    index_finger.y, 
                    image_width, 
                    image_height
                )

                if pixelCoord:
                    cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, COR_DEDO, 5)
                    
                    if check_collision(pixelCoord[0], pixelCoord[1], x_enemy, y_enemy):
                        print(f"Colisão detectada! Score: {score + 1}")
                        x_enemy = random.randint(50,600)
                        y_enemy = random.randint(50,400)
                        score += 1
                        enemy(image)

        cv2.imshow('Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(f"Pontuação final: {score}")
            break

print("Encerrando o jogo...")
video.release()
cv2.destroyAllWindows()




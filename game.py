import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import math
import os
from datetime import datetime

print("Iniciando o jogo...")
print("Carregando bibliotecas...")

draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score = 0
nivel = 1
tempo_inicial = 0
tempo_jogo = 60
alvos_acertados = 0
alvos_necessarios = 5
jogo_iniciado = False
record_file = "recordes.txt"

x_enemy = random.randint(50,600)
y_enemy = random.randint(50,400)

COR_ALVO = (100, 200, 100)
COR_DEDO = (200, 100, 100)
COR_PLACAR = (200, 200, 200)
COR_LANDMARKS = (150, 150, 150)
COR_TEXTO = (255, 255, 255)

def salvar_recorde(nivel, score):
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(record_file, "a", encoding="utf-8") as arquivo:
        arquivo.write(f"Data: {data_hora} - Nivel: {nivel} - Score: {score}\n")

def carregar_ultimo_recorde():
    if not os.path.exists(record_file):
        return 0, 0
    
    try:
        with open(record_file, "r", encoding="utf-8") as arquivo:
            linhas = arquivo.readlines()
            if linhas:
                ultima_linha = linhas[-1]
                partes = ultima_linha.split(" - ")
                nivel = int(partes[1].split(": ")[1])
                score = int(partes[2].split(": ")[1])
                return nivel, score
    except:
        pass
    return 0, 0

def enemy(image):
    global x_enemy, y_enemy
    cv2.circle(image, (x_enemy, y_enemy), 25, COR_ALVO, 5)

def check_collision(x1, y1, x2, y2, threshold=50):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance < threshold

def atualizar_nivel():
    global nivel, alvos_necessarios, tempo_jogo, tempo_inicial
    nivel += 1
    alvos_necessarios += 3
    tempo_jogo = max(30, 60 - (nivel * 5))
    tempo_inicial = time.time()

def mostrar_tela_inicial(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ultimo_nivel, ultimo_score = carregar_ultimo_recorde()
    
    cv2.putText(image, "Pressione ESPACO para comecar", (100,200), font, 1, COR_TEXTO, 2, cv2.LINE_AA)
    cv2.putText(image, "Use seu dedo indicador para tocar nos alvos", (50,250), font, 0.7, COR_TEXTO, 2, cv2.LINE_AA)
    cv2.putText(image, "Pressione Q para sair", (200,300), font, 0.7, COR_TEXTO, 2, cv2.LINE_AA)
    cv2.putText(image, f"Ultimo Recorde - Nivel: {ultimo_nivel} Score: {ultimo_score}", (50,350), font, 0.7, COR_TEXTO, 2, cv2.LINE_AA)

print("Iniciando câmera...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erro: Não foi possível abrir a câmera!")
    exit()

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Iniciando detecção de mãos...")
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    print("Jogo iniciado! Pressione ESPACO para começar.")
    
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

        if not jogo_iniciado:
            mostrar_tela_inicial(image)
            cv2.imshow('Tracking', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):
                jogo_iniciado = True
                tempo_inicial = time.time()
            elif key == ord('q'):
                break
            continue

        tempo_restante = max(0, tempo_jogo - (time.time() - tempo_inicial))
        
        cv2.putText(image, f"Score: {score}", (50,30), font, 1, COR_PLACAR, 2, cv2.LINE_AA)
        cv2.putText(image, f"Nivel: {nivel}", (50,70), font, 1, COR_PLACAR, 2, cv2.LINE_AA)
        cv2.putText(image, f"Tempo: {int(tempo_restante)}s", (50,110), font, 1, COR_PLACAR, 2, cv2.LINE_AA)
        cv2.putText(image, f"Alvos: {alvos_acertados}/{alvos_necessarios}", (50,150), font, 1, COR_PLACAR, 2, cv2.LINE_AA)
        
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
                        alvos_acertados += 1
                        
                        if alvos_acertados >= alvos_necessarios:
                            atualizar_nivel()
                            alvos_acertados = 0
                        
                        enemy(image)

        if tempo_restante <= 0:
            cv2.putText(image, "TEMPO ACABOU!", (200,300), font, 2, (0,0,255), 3, cv2.LINE_AA)
            cv2.imshow('Tracking', image)
            cv2.waitKey(2000)
            salvar_recorde(nivel, score)
            break

        cv2.imshow('Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(f"Pontuação final: {score}")
            print(f"Nível alcançado: {nivel}")
            salvar_recorde(nivel, score)
            break

print("Encerrando o jogo...")
video.release()
cv2.destroyAllWindows()




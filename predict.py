import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os


# Carregar modelo treinado
model = load_model("/mnt/Backup/Programming/Projeto IFAP/Model train/modelLinux.h5")
IMG_SIZE = (224, 224)

# Mapeamento dos índices para classes
class_names = ['A', 'B', 'C','D','E','I','L','M','N','O','R','S','U','V','W']

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Iniciar câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontal para simular espelho
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obter bounding box da mão
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Adicionar margem
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Extrair ROI (Região de Interesse)
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                continue



            # Preprocessar ROI
            roi_resized = cv2.resize(roi, IMG_SIZE)
            roi_preprocessed = preprocess_input(roi_resized.astype(np.float32))
            roi_input = np.expand_dims(roi_preprocessed, axis=0)
            cv2.imshow("Recorte", roi_resized)

            # Predição
            try:
                prediction = model.predict(roi_input)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]
                confidence = prediction[0][class_index]
            except:
                continue

            # Exibir resultado
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_label} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Desenhar landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar imagem
    cv2.imshow("Reconhecimento LIBRAS em Tempo Real", frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar
cap.release()
cv2.destroyAllWindows()
hands.close()
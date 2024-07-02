import os
import cv2
import mediapipe as mp
from fer import FER
import tensorflow as tf

# Suprimir mensagens de log do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inicializar o módulo de detecção facial do mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar o detector de emoções da biblioteca fer
emotion_detector = FER()

# Inicializar o VideoCapture
ip = "http://192.168.18.2:4747/video"
video = cv2.VideoCapture(ip)

if not video.isOpened():
    print("Error: Could not open video stream.")
else:
    try:
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:

            while True:
                check, img = video.read()
                if not check:
                    print("Error: Failed to read frame.")
                    break

                # Converter a imagem para RGB (mediapipe usa RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Realizar a detecção de rostos na imagem
                results = face_detection.process(img_rgb)

                # Verificar se foram encontrados rostos
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(img, detection)
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = img.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                     int(bboxC.width * iw), int(bboxC.height * ih)

                        # Recortar a região do rosto
                        face_img = img[y:y+h, x:x+w]

                        # Detectar emoções no rosto
                        emotions = emotion_detector.detect_emotions(face_img)

                        # Verificar se foram detectadas emoções
                        if emotions:
                            emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                            if emotion in ["happy", "sad"]:
                                emotion_text = "Feliz" if emotion == "happy" else "Triste"
                            else:
                                emotion_text = ""

                            # Escrever a emoção detectada acima do rosto
                            if emotion_text:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img, emotion_text, (x, y-10), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

                # Mostrar a imagem com as detecções
                cv2.imshow("Detecção Facial e Emoções", img)

                # Pressione 'q' para sair do loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")

# Liberar o objeto de captura de vídeo e fechar todas as janelas abertas
video.release()

cv2.destroyAllWindows()

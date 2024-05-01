import cv2
import numpy as np
from rknn.api import RKNN

# Rutas de los archivos
MODEL_FILE = 'yolov5.rknn'
VIDEO_FILE = 'carros.mp4'

# Función para mostrar los rectángulos de detección y etiquetas
def draw_detections(frame, detections):
    for detection in detections:
        class_id, score, x_center, y_center, width, height = detection
        # Convertir coordenadas a esquinas del cuadro delimitador
        x_min = int((x_center - width / 2) * frame.shape[1])
        y_min = int((y_center - height / 2) * frame.shape[0])
        x_max = int((x_center + width / 2) * frame.shape[1])
        y_max = int((y_center + height / 2) * frame.shape[0])

        # Dibujar rectángulo de detección
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Mostrar etiqueta
        label = 'Clase: {} - Confianza: {:.2f}'.format(class_id, score)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    # Inicializar RKNN
    rknn = RKNN()

    # Cargar el modelo RKNN
    print('--> Cargando modelo RKNN')
    ret = rknn.load_rknn(MODEL_FILE)
    if ret != 0:
        print('Error al cargar el modelo RKNN')
        exit(ret)
    print('Hecho')

    # Inicializar captura de video
    cap = cv2.VideoCapture(VIDEO_FILE)

    # Inicializar bucle de procesamiento de video
    while cap.isOpened():
        # Leer el siguiente fotograma
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar el fotograma (puedes necesitar ajustar el tamaño según lo requerido por YOLOv5)
        input_img = cv2.resize(frame, (640, 640))  # Tamaño de entrada de YOLOv5
        input_img = input_img / 255.0  # Normalizar

        # Realizar la inferencia
        outputs = rknn.inference(inputs=[input_img])

        # Procesar las detecciones
        detections = []
        for output in outputs:
            for detection in output:
                class_id, score, x_center, y_center, width, height = detection
                detections.append(detection)

        # Dibujar rectángulos de detección y etiquetas
        draw_detections(frame, detections)

        # Mostrar el fotograma con las detecciones
        cv2.imshow('Frame', frame)
        
        # Salir del bucle al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

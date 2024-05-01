import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Rutas de los archivos
MODEL_FILE = 'yolov5s_relu.onnx'
VIDEO_FILE = 'carros.mp4'

# Función para mostrar los rectángulos de detección y etiquetas
def draw_detections(frame, detections):
    for detection in detections:
        class_id, score, x, y, w, h = detection
        # Dibujar rectángulo de detección
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Mostrar etiqueta
        label = '{}: {:.2f}'.format(class_id, score)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    # Inicializar RKNNLite
    rknn_lite = RKNNLite()

    # Cargar el modelo ONNX
    print('--> Cargando modelo ONNX')
    ret = rknn_lite.load_onnx(MODEL_FILE)
    if ret != 0:
        print('Error al cargar el modelo ONNX')
        exit(ret)
    print('Hecho')

    # Inicializar entorno de ejecución
    print('--> Inicializando entorno de ejecución')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Error al inicializar el entorno de ejecución')
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

        # Preprocesar el fotograma
        input_img = cv2.resize(frame, (640, 640))
        input_img = np.expand_dims(input_img, 0)

        # Realizar la inferencia
        outputs = rknn_lite.inference(inputs=[input_img])

        # Filtrar las detecciones con confianza mayor a 0.5
        detections = []
        for output in outputs:
            for detection in output:
                class_id, score, x, y, w, h = detection
                if score > 0.5:
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
    rknn_lite.release()

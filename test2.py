import cv2
import numpy as np
import platform
from synset_label import labels
from rknnlite.api import RKNNLite

# Rutas de los archivos
MODEL_FILE = 'mobilenet_v2_for_rk3566_rk3568.rknn'
VIDEO_FILE = 'carros.mp4'

# Función para mostrar las mejores 5 predicciones
def show_top5(result):
    output = result[0].reshape(-1)
    # Obtener los índices de los 5 valores más grandes
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = '-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:"{}"\n'.format(
                index, value, labels[index])
        else:
            topi = '-1: 0.0\n'
        top5_str += topi
    print(top5_str)

if __name__ == '__main__':
    # Inicializar RKNNLite
    rknn_lite = RKNNLite()

    # Cargar el modelo RKNN
    print('--> Cargando modelo RKNN')
    ret = rknn_lite.load_rknn(MODEL_FILE)
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

        # Preprocesar el fotograma
        input_img = cv2.resize(frame, (224, 224))
        input_img = np.expand_dims(input_img, 0)
        input_img = np.transpose(input_img, (0, 3, 1, 2))

        # Realizar la inferencia
        outputs = rknn_lite.inference(inputs=[input_img], data_format=['nchw'])

        # Mostrar resultados
        show_top5(outputs)

        # Mostrar el fotograma con las predicciones
        cv2.imshow('Frame', frame)
        
        # Salir del bucle al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    rknn_lite.release()

from typing import Any
from roboflow import Roboflow
import numpy as np
import supervision as sv
import cv2
from time import time
import glob
import os
import json

def data_find(frame, xyxy):
    cx, cy, theta = 0, 0, 0
    t = [0, 0]
    img = frame[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])]
    y = int(xyxy[3])
    x = int(xyxy[0])
    data = []
    img = cv2.GaussianBlur(img, (7, 7), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tet, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contornos, jerarquia = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if contornos:
        m = cv2.moments(contornos[0])
        ## Centroide ###
        if int(m["m00"]) != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            cv2.circle(th, (cx, cy), 5, (0, 255, 0), -1)

    # t , theta = findTip_2(cx , cy , gray)

    #data = [int(cx + x), int(y - cy), int(theta), int(t[0] + x), int(y - t[1])]
    data = [int(cx + x), int(y - cy)]

    return th, data


def plotBox(results, image):
    data = []

    if results.any():
        for result in results:
            frame, res = data_find(image, result)
            if res[0] != 0:
                cv2.circle(image, (res[0], res[1]), 5, (0, 255, 0), -1)
                cv2.circle(image, (res[3], res[4]), 5, (0, 255, 0), -1)

            data.append(res)

    else:
        print("No se detectan objetos")

    return frame, data


def findTip_2(cx, cy, image):
    imCanny = cv2.Canny(image, 166, 171)
    # Obtener las coordenadas de todos los píxeles en la imagen
    coordenadas_blancas = np.column_stack(np.where(imCanny > 0))
    distancias = np.linalg.norm(coordenadas_blancas - [cx, cy], axis=1)
    indice_punto_mas_lejano = np.argmax(distancias)

    farthest_pixel = coordenadas_blancas[indice_punto_mas_lejano]

    dx = farthest_pixel[0] - cx
    dy = farthest_pixel[1] - cy

    angle_rad = np.arctan2(dy, dx)
    angle = np.degrees(angle_rad)

    print(coordenadas_blancas[indice_punto_mas_lejano])

    return farthest_pixel, angle


def plotBox(results, image):
    data = []

    if results.any():
        for result in results:
            frame, res = data_find(image, result)
            if res[0] != 0:
                cv2.circle(image, (res[0], res[1]), 5, (0, 255, 0), -1)
                # cv2.circle(image , (res[3] , res[4]) , 5 , (0 , 255 , 0) , -1)

            data.append(res)

    else:
        print("No se detectan objetos")

    return frame, data


def analisar(image):
    result = model.predict(image, confidence=40, overlap=30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    image = cv2.imread(image)

    annotated_image, res = plotBox(detections.xyxy, image)

    # annotated_image = bounding_box_annotator.annotate(
    # scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(
    # scene=annotated_image, detections=detections, labels=labels)

    # sv.plot_image(image=annotated_image, size=(16, 16))

    return res


##Imagen que se va a analizar###

rf = Roboflow(api_key="o4yu0eDLItzjFtAp5IJE")
project = rf.workspace().project("egg-detection-final")
model = project.version(3).model

ruta_carpeta = "B:/Descargas/Simulacion 3e/Simulacion 3e/Hand-Eye-Data"


imagenes = sorted(glob.glob(os.path.join(ruta_carpeta, '*.jpg')) + glob.glob(os.path.join(ruta_carpeta, '*.png')))


mapeo = []
for imagen in imagenes:

    imagen = imagen.replace('\\', '/')
    image = cv2.imread(imagen)
    detector = analisar(imagen)
    mapeo.append(detector)
   
Mapeo_huevos = "B:/Descargas/Simulacion 3e/Simulacion 3e/Mapeo_Huevos.json"

print(mapeo)

with open(Mapeo_huevos, 'w') as f:
    json.dump(mapeo,f)
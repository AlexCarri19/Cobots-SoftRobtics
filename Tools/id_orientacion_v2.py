import cv2
import numpy as np

def find_tip_blob(th , img):
    contornos , jerarquia = cv2.findContours(th , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    if len(contornos) != 0: 
        for contorno in contornos:
            if len(contorno) >= 100: #Limita el tamaño de los objetos que detecta
                elipse = cv2.fitEllipse(contorno)
                centro , ejes , angulo = elipse

                centro = [int(centro[0]) , int(centro[1])]
                centroP = [centro[0] + 100 , centro[1] + 100]

                longitud = 100
                anguloP = angulo + 90

                x_extremo = int(centro[0] + longitud * np.cos(np.radians(angulo)))
                y_extremo = int(centro[1] + longitud * np.sin(np.radians(angulo)))

                x_1 = int(centro[0] + longitud * np.cos(np.radians(anguloP)))
                y_1 = int(centro[1] + longitud * np.sin(np.radians(anguloP)))

                angulo_horizontal = np.degrees(np.arctan2(y_1 - centro[1], x_1 - centro[0]))
                angulo_horizontal3 = (90 - anguloP) % 360

    return angulo_horizontal3 , centro 

def cercania_orientacion(theta , centro):
    if theta >= 270: 
        if 355< theta < 3 : print("Alineado")
        else: print("Rot sentido del reloj") 

    elif theta < 270: 
        if 175 < theta < 185 : print("Alineado")
        else: print("Rot sentido del reloj") 

    print(centro)


def main():
    capture = cv2.VideoCapture(1)
    print("Main Program")
    while True:
        ret, img = capture.read()

        #Los filtros son importantes "Usa los 3"
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        tet , th = cv2.threshold(gray , 200 , 255 , cv2.THRESH_BINARY_INV)
        th = cv2.bitwise_not(th) #Muy importante, invierte la imagen

        #Define el angulo
        theta , centro = find_tip_blob(th , img)
        cercania_orientacion(theta , centro)

        if cv2.waitKey(1) & 0xFF == ord('l'):
            cv2.destroyAllWindows()
            break

try:
    main()

except KeyboardInterrupt:
    print('KeyboardInterrupt exception is caught')
    cv2.destroyAllWindows()

else:
    print('No exeptions are caught')
import cv2
import numpy as np

def filtro_1 (img):
    mask = cv2.erode(img , None , iterations = 3)
    mask = cv2.dilate(mask , None , iterations = 3)
    res = cv2.bitwise_and(img,img)

    imgBlur = cv2.GaussianBlur(res , (7 , 7) , 1)
    imgGray = cv2.cvtColor(imgBlur , cv2.COLOR_BGR2GRAY)
    imCanny = cv2.Canny(imgGray , 166 , 171)

    contornos , jerarquia = cv2.findContours(imCanny , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    return imCanny , contornos

def filtro_2 (img):
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    tet , th = cv2.threshold(gray , 200 , 255 , cv2.THRESH_BINARY)
    contornos , jerarquia = cv2.findContours(th , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    return th , contornos

def valor_lejano (mat):
    pass

#B:\Documentos\Programas\Bases de datos\Huevos blanco\huevoB_B1_2.jpg
frame = cv2.imread("B:\Documentos\Programas\Bases de datos\Huevos blanco\huevoB_B1_2.jpg")

th , contornos = filtro_2(frame)

m = cv2.moments(contornos[0])
### Centroide ###
cx = int(m["m10"]/m["m00"])
cy = int(m["m01"]/m["m00"])

res = th


cv2.circle(th , (cx , cy) , 5 , (0 , 255 , 0) , -1)


### Orientacion ###
mu20 = m["mu20"]
mu11 = m["mu11"]
mu02 = m["mu02"]

linea = 50

theta = np.degrees(1/2 * np.arctan2(2 * mu11 , mu20 - mu02))

x_final = int(cx + linea * np.cos(theta))
y_final = int(cy + linea * np.cos(theta))

#cv2.line(th , (cx , cy) , (x_final , y_final) , (0 , 0 , 255) , 2)

cv2.imshow("Pruebas" , th)

cv2.waitKey(0)

cv2.destroyAllWindows
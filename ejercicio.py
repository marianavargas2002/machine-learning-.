import numpy as np
import cv2

def linea(lado):
    im = np.zeros((lado, lado, 3), np.uint8)
    centro = ((lado-1)//2)
    contador = 0
    #Rojo
    for i in range(centro-10,centro+11):
        for j in range(centro-10,centro+11):
            im[i,j] = [0,0,255]
    #Verde
    for i in range(centro-8,centro+9):
        for j in range(centro-8,centro+9):
            im[i,j] = [0,255,0]

    #Azul 
    for i in range(centro-6,centro+7):
        for j in range(centro-6,centro+7):
            im[i,j] = [255,0,0]
    #Negro
    for i in range(centro-4,centro+5):
        for j in range(centro-4,centro+5):
            im[i,j] = [0,0,0]
    # cv2.imshow('Image', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()        

    cv2.imwrite('ejercicio.png',im)
linea(101)

 # im[50,50] = [255,255,255]
    # im[50,51] = [255,255,255]
    # im[50,52] = [255,255,255]
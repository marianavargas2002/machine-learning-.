import numpy as np 
import cv2 
# ejercicios de ML


def linea(lado):
    im = np.zeros((lado,lado,3),np.uint8)

    # for i in range(lado):
    #     im[i,i] = [255,255,255]

    cv2.imwrite('ex1.png',im)
    

linea(100)
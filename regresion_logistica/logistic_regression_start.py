"""
Regresión Logística punto de partida

"""

import numpy as np
import matplotlib.pyplot as plt


LR = 0.02
epochs = 6
BATCH_SIZE = 1000
LAMBDA = 1

#mirar que tan preciso es el modelo al final de hacer el modelo
def accuracy(h, y):
    m = y.shape[0]
    h[h>=0.5] = 1
    h[h<0.5] = 0
    c = np.zeros(y.shape)
    c[y==h] = 1
    return c.sum()/m


def shuffle(x,y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)

    x = x[r_indexes]
    y = y[r_indexes]
    return x,y

def sigmoid (prediccion):
    prediccion = 1/1+np.exp(prediccion)
    return prediccion

def main():
    # Cargar dataset
    x = np.load('x.npy')/255
    y = np.load('y.npy')
    
    x, y = shuffle(x,y)
    #falta normalizar los datos 
   
   

    # #mosrar imagen 
    # plt.imshow(x[-1],cmap='gray')
    # plt.show()

    #numero de pixeles
    n = x[0].shape[0]* x[0].shape[1]
    #numero de observaciones
    m = y.shape[0]
    
    #formatear datos como matriz 2D
    x = x.reshape(m,n)

    x_0 = np.ones((m,1))
    x = np.hstack((x_0,x))
    
    #parametros
    theta = np.random.rand(n+1)


    N = len(y)
    # entrenar modelo
    def entrenamiento(x, y, theta, LR, epochs):
        for _ in range(epochs):
            #predicción
            prediccion = sigmoid(np.dot(x, theta))
            error = prediccion - y
            gradient = np.dot(x.T, error) / N
            theta -= LR * gradient
        return theta,prediccion

    
    theta,prediccion = entrenamiento(x, y, theta, LR, epochs)
    print(y)
    print(prediccion)
    acc = accuracy(prediccion, y)
    print("Accuracy:", acc)
    
if __name__ == '__main__': main()
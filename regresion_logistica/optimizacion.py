import numpy as np
import matplotlib.pyplot as plt

LR = 0.1
epochs = 6
BATCH_SIZE = 1000
LAMBDA = 1

def accuracy(h, y):
    m = y.shape[0]
    h[h >= 0.5] = 1
    h[h < 0.5] = 0
    c = np.zeros(y.shape)
    c[y == h] = 1
    return c.sum() / m

def shuffle(x, y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)

    x = x[r_indexes]
    y = y[r_indexes]
    return x, y

def sigmoid(prediccion):
    prediccion = 1 / (1 + np.exp(-prediccion))
    return prediccion

def main():
    # Cargar dataset
    x = np.load('x.npy') / 255
    y = np.load('y.npy')

    x, y = shuffle(x, y)

    # Normalizar datos
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Numero de pixeles
    n = x[0].shape[0] * x[0].shape[1]
    # Numero de observaciones
    m = y.shape[0]

    # Formatear datos como matriz 2D
    x = x.reshape(m, n)
    um_rows, num_columns = x.shape

    x_0 = np.ones((m, 1))
    x = np.hstack((x_0, x))

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    num_validation = int(0.1 * m)
    num_test = int(0.1 * m)
    
    x_training = x[num_validation+num_test:, :]
    x_validation = x[num_test:num_validation+num_test, :]
    x_test = x[:num_test, :]

   
    
    y_validation = y[num_test:num_validation + num_test]
    y_test = y[:num_test]

    # Inicializar parámetros
    theta = np.random.rand(n + 1)

    N = len(y)

    # Entrenar modelo con regularización L2
    def entrenamiento(x, y, theta, LR, epochs):
        costos = []
        acc = []
        for i in range(epochs):
            for batch_start in range (0,len(x_training),BATCH_SIZE):
                x_batch = x[batch_start:batch_start + BATCH_SIZE, :]
                y_batch = y[batch_start:batch_start + BATCH_SIZE]

                prediccion = sigmoid(np.dot(x_batch, theta))
                error = prediccion - y_batch
                gradient = (np.dot(x_batch.T, error) + LAMBDA * theta) / len(y_batch)  # Regularización L2
                
                #excluir el termino bias
                gradient[1:] += LAMBDA * theta[1:] 
                
                theta -= LR * gradient
                costo = (-1/len(y_batch)) * np.sum(y_batch * np.log(prediccion + 0.000001) + (1 - y_batch) * np.log(1 - prediccion + 0.000001))
                costos.append(costo)
                acc.append(accuracy(prediccion, y_batch))
            return theta, prediccion, costos, acc

    theta, prediccion, costos, acc = entrenamiento(x, y, theta, LR, epochs)  

    plt.plot(costos)
    plt.ylabel('Costo')
    plt.title('Costo')
    plt.show()
    plt.title('Accuracy')
    plt.plot(acc)
    plt.show()
    # print(y)
    # print(prediccion)
    aux = accuracy(prediccion,y)
    print("precision del modelo",aux)

    prediccion_validacion = sigmoid(np.dot(x_validation, theta))
    prediccion_test = sigmoid(np.dot(x_test, theta))

    acc_validacion = accuracy(prediccion_validacion, y_validation)
    acc_test = accuracy(prediccion_test, y_test)

    print("Precisión en el conjunto de validación:", acc_validacion)
    print("Precisión en el conjunto de prueba:", acc_test)


  
    

if __name__ == '__main__': main()
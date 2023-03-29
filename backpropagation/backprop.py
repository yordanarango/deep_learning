import mnist_loader_current
import network

training_data, validation_data, test_data = mnist_loader_current.load_data_wrapper() # se leen los datos de entrenamiento, de test y de validación
net                                       = network.Network([784, 100, 10]) # aquí se crea la arquitectura de la red con la primera capa de 784 neuronas (número de pixeles en una imagen), 30 en la segunda y 10 en la de salida
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#DIRECCIÓN DONDE ESTAN GUARDADOS LOS ARCHIVOS
import os
os.chdir(r"C:\Users\lalo_\.spyder-py3\ARCHIVOS DE TRABAJO\RepoClon")
#%%
import mnist_loader
import network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=network.Network([784,30,10])
#%% ENTRENANDO RED 1
net.SGD( training_data, 15, 50, 0.5, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#%% ENTRENANDO RED 2
net.SGD( training_data, 15, 30, 0.1, test_data=test_data)

archivo = open("red_prueba2.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#%% MAS ENTRENAMIENTO
archivo_lectura = open("red_prueba2.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD( training_data, 15, 50, 0.5, test_data=test_data)

archivo = open("red_prueba2.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#%% LEER EL ARCHIVO PARA SU USO
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()



"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from matplotlib.image import imread

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):

    def __init__(self, sizes):
        #Se crea la función de inicialización para construir la red. Entre
        #Se define el numero de layers de la red, la forma y se inicializa z.
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        #Cambio en la inicialización de los pesos
        sigma = 1/self.sizes[0]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [sigma*np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        #Creación de la matriz de momentos
        self.vel_b = [np.zeros(b.shape) for b in self.biases]
        self.vel_w = [np.zeros(w.shape) for w in self.weights]        
        #Definimos el parametro beta
        self.beta = 0.9
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
       # Store all activations and z-values for backpropagation
        activations = [a]
        zs = []
        #Evalua la red en con la información de la imagen
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            zs.append(z)
            #Se procesan todos los layers salvo el ultimo para 
            #activarlo con una soft max
            if i == len(self.weights) - 1:
                a = self.soft_max(z)
            else:
                a = sigmoid(z)
                
            activations.append(a)
        
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        #Escribimos la beta dentro de la función
        beta = self.beta
        
        #Si hay test_data se crean las listas correspondientes
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            
        #Se hace lo mismo que el paso anterior para la test_data
        training_data = list(training_data)
        n = len(training_data)
        
        #Se barajea de forma aleatoria la training data y se crean los mini
        #batches en base a la nueva training data
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            #Se actualizan los minibatches con el nuevo parametro beta
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, beta)
            
            #Si existe una test data se imprime un mensaje que nos deja saber
            #a cuantos datos le atino la red neuronal y en que epoca
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            #si no solo se imprime la epoca
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, beta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
         
        #Primero se crean las matrices de zero para guardar los valores 
        #de las derivadas pesos y los bias  
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #Se hacen las matrices de paso (cambio en los parametros) 
        #para los pesos y los bias
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        #Se crea la función de paso para actualizar los pesos y bias por medio
        #de los nuevos momentos
        self.vel_b = [
            beta*v_b - eta*nb/len(mini_batch) 
            for v_b, nb in zip(self.vel_b, nabla_b)]
        
        self.vel_w = [
            beta*v_w - eta*nw/len(mini_batch) 
            for v_w, nw in zip(self.vel_w, nabla_w)]
        
        #Actualizamos los nuevos pesos en función de los nuevos momentos
        
        self.weights = [w + v_w for w,v_w in zip(self.weights,self.vel_w)]
        
        self.biases = [w + v_b for w,v_b in zip(self.biases,self.vel_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) 
            activation = sigmoid(z)
            activations.append(activation)
        
        #Uso simplificado del soft_max para el output layer
        activations[-1] = self.soft_max(zs[-1])
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """Esta función "evalua" la red neuronal 
        comparando aciertos contra datos totales"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Se crea el test result donde
        #"y" es el dato de testeo y "x" el dato de testeo evaluado por la red.
        return sum(int(x == y) for (x, y) in test_results) #suma las veces que
        #x = y dentro de los test results

    def cost_cross_entropy(self,output_activations , y):
        #Se define la función de costo para ver su cambio.
        epsilon = 1e-9
        return -np.sum(y*np.log(output_activations + epsilon))    

    def cost_derivative(self, output_activations, y):
        #La función regresa la derivada de la función de costo previamente
        #calculada a mano
        return (output_activations-y)
    
    
    def soft_max(self,last_activation):
       #Se define la función soft_max de forma que se evite el overflow
        shifted_z = last_activation - np.max(last_activation) #se define z con 
        #una corrimiento para evitar el overflow
        exp_value = np.exp(shifted_z) #Numerador de la función softmax
        sum_exp_value = np.sum(exp_value) #Denominador del softmax
        return exp_value/sum_exp_value #Salida softmax
    
    def mono_blk(self, image):
        imtest = imread(image) #Leemos nuestra imagen
        imtest=np.reshape(imtest,(784,3)) # La convertimos en vector
        #Convertimos a blanco y negro la imagen:
        lst = []
        for i in imtest:
            pix = i[0]*0.2125+i[1]*0.7174+i[2]*0.0721
            if(pix<125):    
                pix=255. 
            else:
                pix = 0. 
            lst.append(pix)
        imtest=np.array(lst).reshape(28,28) 
        imtest = (imtest/imtest.max())
        return imtest
    
    
    def planar(self, imtest):
        #Esta función se encarga de aplanar la imagen blanco y negro 
        #de entrada para usarse en el feedfoward.
        imtest = self.mono_blk(imtest) #Se convierte a blanco y negro.
        imtest = np.reshape(imtest,(784,1)) #Se aplana
        return imtest

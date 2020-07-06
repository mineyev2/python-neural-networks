import numpy as np
from numpy import linalg as LA

'''
Activation Functions
'''

leaky_slope = 0.01

class Network:
    def __init__(self, layers_dims):
        self.layers_dims = layers_dims
        self.parameters = self.initialize_variables_deep(self.layers_dims)
        self.costs = []
        
    
    '''
    Variable initialization
    '''

    def initialize_variables_deep(self, layers_dims):
        parameters = {}

        L = len(layers_dims)

        input_dims = layers_dims[0]

        for i in range(0, L - 1):
            parameters["W" + str(i + 1)] = np.random.rand(layers_dims[i + 1], layers_dims[i]) / np.sqrt(input_dims)
            parameters["b" + str(i + 1)] = np.zeros((layers_dims[i + 1], 1))

        return parameters

    '''
    Forward Prop
    '''

    def linear_forward(self, W, b, A_prev, activation_function):

        forward_cache = {}

        Z = np.dot(W, A_prev) + b

        # checking strings seems slow
        if(activation_function == "sigmoid"):
            A = sigmoid(Z)
        elif(activation_function == "relu"):
            A = relu(Z)
        elif(activation_function == "leaky"):
            A = leaky_relu(Z)

        return A, Z

    def full_forward_prop(self, X):

        # // is called "floor division", which divides to an integer by getting rid of the remainder
        
        # L here and the L in initialize_parameters_deep are different, the L in here is one smaller and is the number of
        # weight matrices are created for the given neural network
        L = len(self.parameters) // 2

        A_cache = {}
        Z_cache = {}

        A = X
        
        for i in range(0, L - 1):
            A_prev = A
            A, Z = self.linear_forward(self.parameters["W" + str(i + 1)], self.parameters["b" + str(i + 1)], A_prev, "sigmoid")
            A_cache["A" + str(i + 1)] = A
            Z_cache["Z" + str(i + 1)] = Z
        AL, ZL = self.linear_forward(self.parameters["W" + str(L)], self.parameters["b" + str(L)], A, "sigmoid")
        A_cache["A" + str(L)] = AL
        Z_cache["Z" + str(L)] = ZL


        return AL, A_cache, Z_cache

    '''
    Compute Cost
    '''

    # MSE stands for Mean Squared Error and is one of the ways of computing the cost function for a NN
    def MSE_compute(self, AL, Y):

        m = Y.shape[1]

        error_matrix = Y - AL

        # need to sum this to get the actual total output
        cost = 1 / (2 * m) * np.sum((LA.norm(error_matrix, axis=0))
                                    , axis=0)
        return cost
    
    def Logistic_Compute(self, AL, Y):
        
        m = Y.shape[1]
        
        cost = -1 / m * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)), axis=1)
        cost = np.squeeze(cost)
        
        return cost

    '''
    Backward Prop
    '''
    
    def Logistic_derivative(self, AL, Y):
        
        m = Y.shape[1]
        
        
        dZ = AL - Y
        
        return dZ
        

    def MSE_derivative(self, AL, Y):
        
        m = Y.shape[1]
        self.m = m
        
        # commenting out the m since that's what was making my neural network not run 
        return (AL - Y) #/ m
    
    def Cross_Entropy_derivative(self, AL, Y):
        
        dZ = AL - Y
        return dZ

    def linear_backwards(self, dA, Z, A_prev, W, d_activation_function):

        m = A_prev.shape[1]

        if(d_activation_function == 'd_sigmoid'):
            dZ = np.multiply(dA, d_sigmoid(Z))
        elif(d_activation_function == 'd_relu'):
            dZ = np.multiply(dA, d_relu(Z))
        elif(d_activation_function == 'd_leaky'):
            dZ = np.multiply(dA, d_leaky_relu(Z))
        else:
            # for the logistic thing
            dZ = dA

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db


    def full_linear_backwards(self, A_cache, X, Y, Z_cache):

        AL = A_cache["A" + str(len(A_cache))]
        A_prev = A_cache["A" + str(len(A_cache) - 1)]
        ZL = Z_cache["Z" + str(len(Z_cache))]

        # check this
        
        
        dA = self.MSE_derivative(AL, Y)

        L = len(self.parameters) // 2

        dA_prev, dW, db = self.linear_backwards(dA, ZL, A_prev, self.parameters["W" + str(L)], "d_sigmoid")
        '''
        
        #logistic cost function section
        
        dZ = self.Logistic_derivative(AL, Y)

        L = len(self.parameters) // 2

        dA_prev, dW, db = self.linear_backwards(dZ, ZL, A_prev, self.parameters["W" + str(L)], "random")
        '''
        grads = {}

        grads["dW" + str(L)] = dW
        grads["db" + str(L)] = db

        for i in range(L - 1, 1, -1):
            dA_prev, dW, db = self.linear_backwards(dA_prev, Z_cache["Z" + str(i)], A_cache["A" + str(i - 1)], self.parameters["W" + str(i)], 'd_sigmoid')
            grads["dW" + str(i)] = dW
            grads["db" + str(i)] = db

        dA_prev, dW, db = self.linear_backwards(dA_prev, Z_cache["Z1"], X, self.parameters["W1"], 'd_sigmoid')
        grads["dW1"] = dW
        grads["db1"] = db
        return grads


    '''
    Update Parameters
    '''

    def update_parameters(self, learning_rate):
        L = len(self.parameters) // 2
        
        for i in range(0, L):
            self.parameters["W" + str(i + 1)] -= learning_rate * self.grads["dW" + str(i + 1)]
            self.parameters["b" + str(i + 1)] -= learning_rate * self.grads["db" + str(i + 1)]

    '''
    Model Training
    '''

    def train_model(self, X, Y, iterations, learning_rate):
        
        if(iterations == 0):
            AL, A_cache, Z_cache = self.full_forward_prop(X)
            self.train_AL = AL
            self.train_estimates, self.train_vectorized_estimates = self.estimate(AL)
            return
        
        self.X = X
        self.Y = Y
        
        for i in range(0, iterations):
            AL, A_cache, Z_cache = self.full_forward_prop(X)
            
            self.grads = self.full_linear_backwards(A_cache, X, Y, Z_cache)
            self.update_parameters(learning_rate)
            
            if((i + 1) % 100 == 0):
                cost = self.MSE_compute(AL, Y)
                print("cost after " + str(i + 1) + " iterations: " + str(cost))
                self.costs.append(cost)
        self.train_AL = AL
        self.train_A_cache = A_cache
        self.train_Z_cache = Z_cache
        self.train_estimates, self.train_vectorized_estimates = self.estimate(AL)
        
    def test_model(self, X, Y):
        AL, A_cache, Z_cache = self.full_forward_prop(X)
        
        self.test_estimates, self.test_vectorized_estimates = self.estimate(AL)
        
        self.test_accuracy = self.compute_accuracy(Y)
        
        self.test_AL = AL
        
        cost = self.MSE_compute(AL, Y)
        print("cost of test set: " + str(cost))
        
    def predict(self, X):
        AL, A_cache, Z_cache = self.full_forward_prop(X)
        
        self.predict_AL = AL
        
        self.test_estimates, self.test_vectorized_estimates = self.estimate(AL)
        
        
    def estimate(self, AL):
        estimates = np.argmax(AL, axis=0)
        #print(estimates)
        
        vectorized_estimates = np.zeros(AL.shape)
        
        for i in range(0, len(estimates)):
            vectorized_estimates[estimates[i]][i] = 1
            
        #vectorized_estimates = np.add(vecto)
        #print(vectorized_estimates)
        
        return estimates, vectorized_estimates
    
    def compute_accuracy(self, Y):
        #only for the test set
        print(Y.shape)
        return (Y.shape[1] - np.sum((np.sum(np.abs(np.subtract(self.test_vectorized_estimates, Y)), axis=0) / 2), axis=0)) / Y.shape[1]

'''
Activation functions
'''

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# I think in the notes they did something smarter than this
def d_sigmoid(x):
    ds = sigmoid(x) * (1 - sigmoid(x))
    return ds

def relu(x):
    '''
    x[x<0] = 0
    return x
    '''
    r = np.maximum(0, x)
    return r

def d_relu(x):
    '''
    x[x <= 0] = 0
    x[x > 1] = 1
    return x
    '''
    dr = np.greater(x, 0).astype(int)
    return dr
    
def leaky_relu(x):
    '''
    x[x<0] *= leaky_slope
    return x
    '''
    lr = np.maximum(x, x * leaky_slope)
    return lr
    
# not working right now
def d_leaky_relu(x):
    '''
    x[x < 0] = leaky_slope
    x[x > 1] = 1
    return x
    '''
    dlr = np.ones_like(x)
    dlr[x < 0] = leaky_slope
    
    return dlr

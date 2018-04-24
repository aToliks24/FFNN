# coding=utf-8
import numpy as np
import idx2numpy
import time
np.random.seed(1)
eps = 0.00001
def load_data_set(numbers_classes):
    """
    
    :param numbers_classes: list of size 2, with the numbers we want to select from the dataset 
            usage example load_data_set([3,8]) where the label for 3 is 0 and the label for 8 is 1
    :return: (x_train, y_train), (x_test, y_test) --- X – the data, numpy array of shape (input size (features), number of examples)
    """

    # train

    x_train = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')  # shape ((m_train, 28, 28)
    x_train = x_train.reshape(x_train.shape[0], -1)  # shape (m_train,784)
    print("    x_train = x_train.reshape(x_train.shape[0], -1).T:   ", x_train.shape)

    y_train = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
    #y_train = np.expand_dims(y_train, axis=0)
    print("np.expand_dims(y_train, axis=0):    ", y_train.shape)

    mask = np.vectorize(lambda t: True if t in numbers_classes else False)
    train_mask = mask(y_train)
    print("train_mask = mask(y_train):     ", train_mask.shape)
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    class_to_binary = np.vectorize(lambda t: 0 if t == numbers_classes[0] else 1)
    y_train = class_to_binary(y_train)

    # test

    x_test = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')  # shape ((m_test, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], -1)  # shape (m_test, 784 )

    y_test = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')
    #y_test = np.expand_dims(y_test, axis=0)

    test_mask = mask(y_test)

    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    y_test = class_to_binary(y_test)

    # shapes: ((784, m_train), (1, m_train)) ((784, m_test), (1, m_test))
    return (x_train.T, np.expand_dims(y_train, axis=0)), (x_test.T, np.expand_dims(y_test, axis=0))




def initialize_parameters(layer_dims):
    """
    input: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output sigmoid)
    output: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    init_params = {}
    for i, layer in enumerate(layer_dims[1:]):
        #print(i)
        init_params['W'+str(i+1)] = np.random.rand(layer_dims[i+1],layer_dims[i])*0.001 # should we use specific range for this initialization?
        init_params['b'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],1))

    return init_params

def linear_forward(A, W, b):
    """
    Description:
    Implement the linear part of a layer's forward propagation.
    input:
    A – the activations of the previous layer
    W – the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    B – the bias vector of the current layer (of shape [size of current layer, 1])
    Output:
    Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
    linear_cache – a dictionary containing A, W, b and Z (stored for making the backpropagation easier to compute)
    """
    Z = np.dot(W,A) + b # broadcasting
    linear_cache = {"A": A, "W": W, "b": b}

    return Z, linear_cache


def sigmoid(Z):
    """
    Input:
    Z – the linear component of the activation function
    Output:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = 1.0 / (1 + np.exp(-1.0*Z))
    activation_cache = {"Z": Z}

    return A, activation_cache


def relu(Z):
    """
    Input:
    Z – the linear component of the activation function
    Output:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0,Z)
    activation_cache = {"Z": Z}

    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    Description:
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Input:
    A_prev – activations of the previous layer
    W – the weights matrix of the current layer
    B – the bias vector of the current layer
    Activation – the activation function to be used (a string, either “sigmoid” or “relu”)
    Output:
    A – the activations of the current layer
    linear_cache – the dictionary generated by the linear_forward function
    """

    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
        return A, {**linear_cache, **activation_cache}
    elif activation == "relu":
        A, activation_cache = relu(Z)
        return A, {**linear_cache, **activation_cache}


def L_model_forward(X, parameters):
    """
    Description: Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Input:
    X – the data, numpy array of shape (input size, number of examples)
    parameters – the initialized W and b parameters of each layer
    Output:
    AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    """
    caches = []
    L = int(len(parameters) / 2)  # parameters is a dictionary which holds Wl and bl for each layer l
    A = X
    for i in range(1,L):
        A, cache = linear_activation_forward(A,parameters['W'+str(i)], parameters['b'+str(i)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    Description: Implement the cost function defined by equation
    𝑐𝑜𝑠𝑡=−1𝑚∗Σ[(𝑦𝑖∗log(𝐴𝐿))+((1−𝑦𝑖)∗(1−𝐴𝐿))]𝑚1 (see the slides of the first lecture for additional information if needed).
    Input:
    AL – probability vector corresponding to your label predictions, shape (1, number of examples)
    Y – the labels vector (i.e. the ground truth)
    Output:
    cost – the cross-entropy cost
    """
    #TODO verify len(Y): Y is (m,1) or(1,m)?? Answer: Y is (1,m)
    cost = -1.0/Y.shape[1] * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    return cost



def linear_activation_backward(dA, cache, activation):
    """
    Description:
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.
    ￼￼￼￼￼￼￼￼￼
    Input:
    dA – post activation gradient of the current layer
    cache – contains both the linear cache and the activations cache
    Output:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b

    """
    if activation == "sigmoid":
        dZ=sigmoid_backward(dA,cache)
    elif activation == "relu":
        dZ=relu_backward(dA,cache)
    else:
        raise NotImplemented("No Such activation function")
    dA_prev, dW, db=linear_backward(dZ,cache)
    return dA_prev, dW, db


def linear_backward(dZ,cache):
    """
    Description :Implements the linear part of the backward propagation process for a single layer

    Input:
    dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    dA_prev - Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW - Gradient of the cost with respect to W (current layer l), same shape as W
    db - Gradient of the cost with respect to b (current layer l), same shape as b
    """
    dA_prev=np.dot(cache["W"].T,dZ)
    dW=np.dot( dZ,cache['A'].transpose())
    db=np.average(dZ, axis=1).reshape((dZ.shape[0], 1))
    return dA_prev,dW,db



def relu_backward (dA, activation_cache):
    """
    Description:
    Implements backward propagation for a ReLU unit
    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)
    Output:
    dZ – gradient of the cost with respect to Z
    """
    gz=activation_cache["Z"].copy()
    gz[gz<0] =0
    gz[gz >=0] =1
    dZ=np.multiply(dA,gz)
    return  dZ


def sigmoid_backward (dA, activation_cache):
    """
    Description:
    Implements backward propagation for a sigmoid unit
    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)
    Output:
    dZ – gradient of the cost with respect to Z
    """
    a,_=sigmoid(activation_cache["Z"])
    gz=np.multiply( a,(1-a))
    dZ = np.multiply(dA, gz)
    return dZ


def L_model_backward(AL, Y, caches):
    """
    Description:
    Implement the backward propagation process for the entire network.
    Some comments:
    - The backpropagation for the Sigmoid should be done separately (because there is only one like it), and the process for the ReLU layers should be done in a loop
    - The derivative for the output of the softmax layer can be calculated using: dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    Input:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y – the true labels vector (the “ground truth” – true classifications)
    Caches – list of caches containing for each layer: a) the linear cache; b) the activation cache
    Output:
    Grads – a dictionary with the gradients
    grads["dA" + str(l)] = ... grads["dW" + str(l)] = ... grads["db" + str(l)] = ...

    """
    dAL= -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    n_layers=len(caches)
    dA_prev, dW, db=linear_activation_backward(dAL,{"W":caches[-1]["W"],"Z":caches[-1]["Z"],"A":caches[-1]["A"]},'sigmoid')
    grads={"dW"+str(n_layers):dW,"db"+str(n_layers):db}
    for i in range (n_layers-2,-1,-1):
        dA_prev, dW, db = linear_activation_backward(dA_prev, {"W":caches[i]["W"],"Z":caches[i]["Z"],"A":caches[i]["A"]}, 'relu')
        grads.update({"dW"+str(i+1):dW,"db"+str(i+1):db})
    return grads


def Update_parameters(parameters, grads, learning_rate):
    """
    Description:
    Updates parameters using gradient descent
    Input:
    parameters – a python dictionary containing the DNN architecture’s parameters
    grads – a python dictionary containing the gradients (generated by L_model_backward)
    learning_rate – the learning rate used to update the parameters (the “alpha”)
    Output:
    parameters – the updated values of the parameters object provided as input
    """

    for l_num in range(1,int(len(parameters)/2+1)):
        parameters["W"+str(l_num)]-=(grads["dW"+str(l_num)]*learning_rate)
        parameters["b"+str(l_num)] -= (grads["db"+str(l_num)] * learning_rate)
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations,verbose=True):
    """
    Description:
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function, and the final layer will apply the sigmoid activation function. The network should only address binary classification.
    Hint: the function should use the earlier functions in the following order: initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters
    Input:
    X – the input data, a numpy array of shape (height*width , number_of_examples) Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
    Y – the “real” labels of the data, a vector of shape (1, number of examples) Layer_dims – a list containing the dimensions of each layer, including the input
    verbose - set True to print the costs over iterations
    Output:
    parameters – the parameters learnt by the system during the training (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """
    parameters=initialize_parameters(layers_dims)
    costs=[]
    for i in range(num_iterations):
        t1=time.time()
        AL, caches=L_model_forward(X,parameters)
        AL[AL == 0] = eps
        AL[AL == 1] -= eps
        cost=compute_cost(AL, Y)
        costs.append(cost)
        grads=L_model_backward(AL,Y,caches)
        parameters=Update_parameters(parameters,grads,learning_rate)
        if (verbose):
            delta=(time.time()-t1)/60
            print("Iteration {}: {:.5} seconds ,Train Cost: {:.5}".format(i,delta,cost))
    return parameters,costs


def Predict(X, Y, parameters):
    """
    Description:
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.
    Input:
    X – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (1, number of examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters
    Output:
    accuracy – the accuracy measure of the neural net on the provided data
    """
    y_pred,_= L_model_forward(X,parameters)
    y_pred[y_pred>=0.5]=1
    y_pred[y_pred < 0.5] = 0
    acc=0.0
    for i,yp in enumerate(y_pred[0]):
        if yp==Y[0][i]:
            acc+=1
    print(str(acc/len(Y[0])))
    return acc/len(Y[0])



def test_forward():
    init = initialize_parameters([10,3,4,5,6,2,1])
    # A = np.expand_dims(np.array([1,1,1,1,1,1,1,1,1,1]), axis=1)
    A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    print ('W1 shape:', init['W1'].shape)
    print (init['W1'])
    print('A shape:', A.shape)
    print (A)
    print('W*A:', np.dot(init['W1'], A), np.dot(init['W1'], A).shape)
    print('b1 shape:', init['b1'].shape)
    print(init['b1'])

    print (init['W1'])
    A_new, cash = linear_forward(A, init['W1'], init['b1'])
    print (A_new)

    print ("-------------------")
    print ('W2 shape:', init['W2'].shape)
    print (init['W2'])
    print('A_new shape:', A_new.shape)
    print (A_new)
    print('b2 shape:', init['b2'].shape)
    print(init['b2'])
    print(linear_forward(A_new, init['W2'], init['b2']))

    print (sigmoid(np.array([[-100,-100,-100],[100,100,100],[1,1,1]])))
    print (relu(np.array([[0.1,0.1,0.1],[-2,-2,-2],[1,1,1]])))

    (x_train, y_train), (x_test, y_test)  = load_data_set([1,2])
    print ((x_train.shape, y_train.shape), (x_test.shape, y_test.shape))

def test_backward():
    (x_train, y_train), (x_test, y_test) = load_data_set([1, 2])
    print("X_train shape: {}".format(x_train.shape))
    params = initialize_parameters([x_train.shape[0], 3,1])
    AL, caches=L_model_forward(x_train,params)
    grads = L_model_backward(AL, y_train, caches)
    print(caches)
    print(grads)

if __name__ == '__main__':
    # test_forward()
    #test_backward()
    (x_train, y_train), (x_test, y_test) = load_data_set([1, 2])
    #x_train=np.divide(x_train,255)
    #x_test=np.divide(x_test,255)
    parameters, costs=L_layer_model(x_train,y_train,[x_train.shape[0],20,7,5,1],0.009,500)
    Predict(x_test,y_test,parameters)

    k=0

import numpy as np


def initialize_parameters(layer_dims):
    """"
    input: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output sigmoid)
    output: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """""
    init_params = {}
    for i, layer in enumerate(layer_dims[1:1]):
        init_params['W'+str(i)] = np.random.rand(layer_dims[i],layer_dims[i-1]) # should we use specific range for this initializttion?
        init_params['b'+str(i)] = np.zeros(shape=(layer_dims[i],1))

    return init_params


# Old code
# class ActivationFunction(object):
#     def activate(self,X,W,b):
#         pass
#     def inv_activate(self,X,W,b):
#         pass
#
#
#
# class LinearActivation(ActivationFunction):
#     def activate(self,X,W,b):
#         return np.transpose(W)*X+b
#
#     def inv_activate(self,X,W,b):
#         return (X-b)/np.transpose(W)
#
#
# class Perceptron(object):
#     _activations = {'linear': LinearActivation}
#
#     def __init__(self,n_inputs,weights=None,activation='linear',seed=None):
#         np.random.seed(seed)
#         self._activation=self._activations[activation]
#         self._inputs=n_inputs
#         self._weights=np.random.rand(n_inputs,1)
#         if weights!=None:
#             self._weights=weights
#
#     def update_weights(self, new_weights):
#         self._weights=new_weights






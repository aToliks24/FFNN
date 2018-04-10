import numpy as np


class ActivationFunction(object):
    def activate(self,X,W,b):
        pass
    def inv_activate(self,X,W,b):
        pass



class LinearActivation(ActivationFunction):
    def activate(self,X,W,b):
        return np.transpose(W)*X+b

    def inv_activate(self,X,W,b):
        return (X-b)/np.transpose(W)


class Perceptron(object):
    _activations = {'linear': LinearActivation}

    def __init__(self,n_inputs,weights=None,activation='linear',seed=None):
        np.random.seed(seed)
        self._activation=self._activations[activation]
        self._inputs=n_inputs
        self._weights=np.random.rand(n_inputs,1)
        if weights!=None:
            self._weights=weights

    def update_weights(self, new_weights):
        self._weights=new_weights






import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['W2'] = np.random.randn(num_filters * (H // 2) * (W //2), hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, affine1_cache = affine_relu_forward(out, W2, b2)
        scores, affine2_cache = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        assert np.all(np.isfinite(loss)), ("Softmax loss contains NaN or infinity: {}".format(loss))
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2 + np.sum(self.params['W3']**2)))
        dx, dw, db = affine_backward(dx, affine2_cache)
        grads['W3'] = dw + (self.reg * self.params['W3'])
        grads['b3'] = db
        dx, dw, db = affine_relu_backward(dx, affine1_cache)
        grads['W2'] = dw + (self.reg * self.params['W2'])
        grads['b2'] = db
        dx, dw, db = conv_relu_pool_backward(dx, pool_cache)
        grads['W1'] = dw + (self.reg * self.params['W1'])
        grads['b1'] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class MyConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - batchnorm - relu - 2x2 max pool -
    affine - batchnorm - relu - dropout
    affine  - batchnorm - relu - dropout
    affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, loss_fn='softmax', dropout=0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.loss_fn = loss_fn
        self.dropout = dropout

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['W2'] = np.random.randn(num_filters * (H // 2) * (W //2), hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, hidden_dim) * weight_scale
        self.params['W4'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['b4'] = np.zeros(num_classes)

        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)
        self.params['gamma3'] = np.ones(hidden_dim)
        self.params['beta3'] = np.zeros(hidden_dim)
        self.bn_params = [{'mode':'train'}, {'mode':'train'}, {'mode':'train'}]
        self.dropout_param = {}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
        self.dropout_param['mode'] = mode
        self.dropout_param['p'] = self.dropout

        out, pool_cache = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, self.bn_params[0])
        out, affine1_cache = affine_bn_relu_forward(out, W2, b2, gamma2, beta2, self.bn_params[1])
        out, dropout1_cache = dropout_forward(out, self.dropout_param)
        out, affine2_cache = affine_bn_relu_forward(out, W3, b3, gamma3, beta3, self.bn_params[2])
        out, dropout2_cache = dropout_forward(out, self.dropout_param)
        scores, affine3_cache = affine_forward(out, W4, b4)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        assert np.all(np.isfinite(loss)), ("Softmax loss contains NaN or infinity: {}".format(loss))
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2 + np.sum(self.params['W3']**2) + np.sum(self.params['W4']**2)))
        dx, dw, db = affine_backward(dx, affine3_cache)
        grads['W4'] = dw + (self.reg * self.params['W4'])
        grads['b4'] = db

        dx = dropout_backward(dx, dropout2_cache)

        dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, affine2_cache)
        grads['gamma3'] = dgamma
        grads['beta3'] = dbeta
        grads['W3'] = dw + (self.reg * self.params['W3'])
        grads['b3'] = db

        dx = dropout_backward(dx, dropout1_cache)

        dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, affine1_cache)
        grads['gamma2'] = dgamma
        grads['beta2'] = dbeta
        grads['W2'] = dw + (self.reg * self.params['W2'])
        grads['b2'] = db

        dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dx, pool_cache)
        grads['gamma1'] = dgamma
        grads['beta1'] = dbeta
        grads['W1'] = dw + (self.reg * self.params['W1'])
        grads['b1'] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads



pass

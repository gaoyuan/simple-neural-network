from activations import Sigmoid
from objectives import SquareLoss
from optimizers import Adam
from random import uniform
from utils import Queue

class Connection(object):
    """Establish the connection from one neuron to another. The connection strength/weight is captured by w.
    
    :param src: the source neuron
    :param tar: the target neuron
    :param w: the connection weight
    :type w: float
    
    """
    def __init__(self, src, tar, w):
        self.src = src
        self.tar = tar
        self.w = w
        self.src.outputs.append(self)
        self.tar.inputs.append(self)

class Neuron(object):
    """Class representing a neuron. Its connection to other neurons are defined as a set of input and output connections.

    :param activation: the activation function associated with this neuron

    """
    def __init__(self, activation=Sigmoid):
        self.inputs = []
        self.outputs = []
        self.activation = activation
        self.value = None
        self.error = None

    def get_value(self):
        """Calculate the output of the neuron."""
        if self.value is not None:
            return self.value
        pre_activation = sum([connection.w * connection.src.get_value() for connection in self.inputs])
        self.value = self.activation.f(pre_activation)
        return self.value

    def get_error(self):
        """Calculate the error of the neuron.""" 
        if self.error is not None:
            return self.error
        self.error = sum([connection.w * connection.tar.get_error() for connection in self.outputs])
        return self.error

    def update_weight(self, opt):
        """Update the weights of the input connections, and call the update function for all the output connections.
        
        :param learning_rate: the weight is updated by learning_rate * negative gradient

        """
        assert self.value is not None and self.error is not None
        pre_activation = sum([connection.src.get_value() for connection in self.inputs])
        for connection in self.inputs:
            # for minimization problem, go in the direction of negative gradient
            # connection.w -= learning_rate * self.error * connection.src.value * self.activation.df(pre_activation)
            connection.w -= opt.delta(self.error * connection.src.value * self.value * (1 - self.value))
        for connection in self.outputs:
            connection.tar.update_weight(opt)

    def clear(self):
        """Clear cached values stored in this neuron."""
        self.value = self.error = None
        for connection in self.outputs:
            connection.tar.clear()

class ConstNeuron(Neuron):
    """A neuron with constant value, often used to define a bias neuron."""
    def __init__(self, value):
        self.inputs = []
        self.outputs = []
        self.value = value

    def get_value(self):
        return self.value
    
    def get_error(self):
        raise RuntimeError('get_error() should not be called for ConstNeuron!')

    def update_weight(self):
        raise RuntimeError('update_weight() should not be called for ConstNeuron!')

    def clear(self):
        raise RuntimeError('clear() should not be called for ConstNeuron!')

class NeuronWithBias(Neuron):
    """Neuron with a constant neuron of value 1 attached to it."""
    def __init__(self, activation=Sigmoid):
        super(NeuronWithBias, self).__init__(activation)
        self.inputs.append(Connection(ConstNeuron(1), self, uniform(0,1)))

class LayeredNN(object):
    """A neural network. It consists of a sequence of layers, where the first layer is a set input neurons, and the last layer a set of output neurons.
    The layers in between are hidden layers. Neurons between the layers are fully connected.

    :param layers: a set of integers showing the number of neurons in each layer
    :param loss: loss function
    :param activation: activation function for all neurons
    :param optimizer: optimizer to use for training
    :param max_iter: maximum number of iterations to stop

    """
    def __init__(self, layers, **kwargs):
        self.loss = kwargs.get('loss', SquareLoss)
        self.activation = kwargs.get('activation', Sigmoid)
        self.optimizer = kwargs.get('optimizer', Adam())
        self.max_iter = kwargs.get('max_iter', 10000)
        self.initialize_network(layers)

    def initialize_network(self, layers):
        """Intialize the neuron network. This will initialize three internal variables:
        1. nlayers -- the number of layers
        2. input_layer -- a list of input neurons
        3. output_layer -- a list of output neurons
        
        :param layers: a tuple/list of integers showing the number of neurons in each layer

        """
        self.nlayers = len(layers)
        assert self.nlayers >= 2 # at least an input layer and an output layer
        previous_layer = []
        for layer, num_neurons in enumerate(layers):
            current_layer = [NeuronWithBias(self.activation) for _ in range(num_neurons)]
            # connect the neurons
            for input_neuron in previous_layer:
                for output_neuron in current_layer:
                    Connection(input_neuron, output_neuron, uniform(0, 1))
            if layer == 0:
                self.input_layer = current_layer
            previous_layer = current_layer
        self.output_layer = current_layer

    def _clear_cache(self):
        """Clear temporary cached values in the network neurons."""
        for neuron in self.input_layer:
            neuron.clear()

    def print_weights(self):
        q = Queue()
        layer = 0
        for neuron in self.input_layer:
            q.put((neuron, layer + 1))
        while not q.empty():
            neuron, cur_layer = q.pop()
            if cur_layer > layer and cur_layer < self.nlayers:
                print "\nlayer %d:" % cur_layer,
                layer = cur_layer
                for connection in neuron.outputs:
                    q.put((connection.tar, layer + 1))
            for connection in neuron.outputs:
                print connection.w,
        print # print new line

    def forward(self, input_values):
        """Forward pass given the input values.

        :param input_values: input values to the neuron network
        
        :returns: a list of values for output neurons`:
        """
        assert len(input_values) == len(self.input_layer)
        self._clear_cache()
        for neuron, input_value in zip(self.input_layer, input_values):
            neuron.value = input_value
        return [neuron.get_value() for neuron in self.output_layer]

    def backward(self, input_values, target_values):
        """Backward pass given input and target values.

        :param input_values: input values for the neuron network
        :param target_values: target output values for the neuron network
        :param learning_rate: learning rate for the weight update

        """
        assert len(target_values) == len(self.output_layer)
        output_values = self.forward(input_values)
        for neuron, output, target in zip(self.output_layer, output_values, target_values):
            neuron.error = self.loss.df(output, target)
        for neuron in self.input_layer:
            neuron.get_error()
        for neuron in self.input_layer:
            neuron.update_weight(self.optimizer)

    def train(self, data):
        for i in range(self.max_iter):
            for input_values, target_values in data:
                self.backward(input_values, target_values)
            err = 0
            for input_values, target_values in data:
                predictions = self.forward(input_values)
                for pred, target in zip(predictions, target_values):
                    err += self.loss.f(pred, target)
            print i, err


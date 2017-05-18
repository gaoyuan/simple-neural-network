from math import exp
from functions import Differentiable

class Sigmoid(Differentiable):
    """The Sigmoid activation function."""
    @staticmethod
    def f(x):
        return 1.0 / (1.0 + exp(-x))
    @staticmethod
    def df(x):
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))

class ReLU(Differentiable):
    """The ReLU activation function."""
    @staticmethod
    def f(x):
        return x if x > 0 else 0
    @staticmethod
    def df(x):
        return 1 if x > 0 else 0

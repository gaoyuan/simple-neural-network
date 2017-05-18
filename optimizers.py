from math import sqrt

class Optimizer(object):
    """An optimizer takes a gradient and produces a step."""
    def delta(self, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, eta = 0.001):
        self._eta = eta 
        self._counter = 0 
    def delta(self, grad):
        self._counter += 1
        return self._eta / sqrt(self._counter) * grad

class Momentum(Optimizer):
    def __init__(self, eta = 0.001, gamma = 0.9):
        self._eta = eta 
        self._gamma = gamma
        self._counter = 0 
        self._grad = None

    def delta(self, grad):
        self._counter += 1
        if self._counter == 1:
            self._grad = self._eta * grad
        else:
            self._grad = self._gamma * self._grad + self._eta / sqrt(self._counter) * grad
        return self._grad

class AGD(Optimizer):
    def __init__(self, eta = 0.001, gamma = 0.9):
        self._eta = eta 
        self._gamma = gamma
        self._counter = 0 
        self._grad = None
    
    def delta(self, grad):
        self._counter += 1
        if self._counter == 1:
            self._grad = self._eta * grad
            return self._eta * grad
        old_grad = self._grad
        self._grad = self._gamma * self._grad + self._eta / sqrt(self._counter) * grad
        return self._grad + self._gamma * old_grad

class Ada(Optimizer):
    def __init__(self, eta = 0.001):
        self._eta = eta 
        self._eps = 1e-8
        self._normsq = 0   
 
    def delta(self, grad):
        self._normsq += grad ** 2
        return self._eta / sqrt(self._normsq + self._eps) * grad

class Adadelta(Optimizer):
    def __init__(self, gamma = 0.9):
        self._eps = 1e-8
        self._gamma = gamma
        self._gradsq = 0
        self._deltasq = 0

    def delta(self, grad):
        self._gradsq = self._gamma * self._gradsq + (1 - self._gamma) * grad ** 2
        _delta = sqrt(self._deltasq + self._eps) / sqrt(self._gradsq + self._eps) * grad
        self._deltasq = self._gamma * self._deltasq + (1 - self._gamma) * _delta ** 2
        return _delta

class RMSprop(Optimizer):
    def __init__(self, eta = 0.001, gamma = 0.9):
        self._eta = eta
        self._gamma = gamma
        self._eps = 1e-8
        self._gradsq = 0

    def delta(self, grad):
        self._gradsq = self._gamma * self._gradsq + (1 - self._gamma) * grad ** 2
        return self._eta / sqrt(self._gradsq + self._eps) * grad

class Adam(Optimizer):
    def __init__(self, eta = 0.001, beta1 = 0.9, beta2 = 0.999):
        self._eta = eta
        self._eps = 1e-8
        self._beta1 = beta1
        self._beta2 = beta2
        self._grad = 0
        self._gradsq = 0
        self._counter = 0

    def delta(self, grad):
        self._counter += 1
        self._grad = self._beta1 * self._grad + (1 - self._beta1) * grad
        self._gradsq = self._beta2 * self._gradsq + (1 - self._beta2) * grad ** 2
        return self._eta / sqrt(self._gradsq/(1 - self._beta2**self._counter) + self._eps) * self._grad/(1 - self._beta1**self._counter)


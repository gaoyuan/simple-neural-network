class Function(object):
    @staticmethod
    def f(x):
        """Function value at a point x."""
        raise NotImplementedError

class Differentiable(Function):
    @staticmethod
    def df(x):
        """Derivative of the function at point x."""
        raise NotImplementedError

class Function2D(object):
    @staticmethod
    def f(x, y):
        """Function value at point (x,y)."""
        raise NotImplementedError



from functions import Function2D

class AbsoluteLoss(Function2D):
    """The Square loss function."""
    @staticmethod
    def f(x, y):
        return abs(x - y)
    
    @staticmethod
    def df(x, y):
        """Derivative with respect to x."""
        return x if x > y else -x

class SquareLoss(Function2D):
    """The Square loss function."""
    @staticmethod
    def f(x, y):
        return 0.5 * (x - y)**2
    
    @staticmethod
    def df(x, y):
        """Derivative with respect to x."""
        return x - y

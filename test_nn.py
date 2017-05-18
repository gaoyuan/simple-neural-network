from nn import Connection, Neuron, LayeredNN
import unittest

class TestConnection(unittest.TestCase):
    def test_connection_init(self):
        n1 = Neuron()
        n2 = Neuron()
        c = Connection(n1, n2, 1)
        self.assertEqual(c.w, 1)
        self.assertEqual(c.src, n1)
        self.assertEqual(c.tar, n2)
        self.assertIn(c, n1.outputs)
        self.assertIn(c, n2.inputs)

class TestNeuron(unittest.TestCase):

    def test_get_value(self):
        n1 = Neuron()
        n2 = Neuron()
        c = Connection(n1, n2, 1.5)
        n1.value = 2.0
        self.assertEqual(n1.get_value(), 2.0)
        self.assertEqual(n2.get_value(), n2.activation.f(3.0)) 

    def test_get_error(self):
        n1 = Neuron()
        n2 = Neuron()
        c = Connection(n1, n2, 1.5)
        n2.error = 2.0
        self.assertEqual(n2.get_error(), 2.0)
        self.assertEqual(n1.get_error(), 3.0) 

if __name__ == '__main__':
    unittest.main()    

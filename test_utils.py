from utils import Queue
import unittest

class TestQueue(unittest.TestCase):

    def test_empty(self):
        self.assertTrue(Queue().empty())

    def test_put_1(self):
        q = Queue()
        q.put(1)
        self.assertEqual(q.top(), 1)

    def test_put_2(self):
        q = Queue()
        q.put(2)
        self.assertFalse(q.empty())
        q.put(1)
        self.assertEqual(q.top(), 2)

    def test_pop_1(self):
        q = Queue()
        q.put(1)
        self.assertEqual(q.pop(), 1)
        self.assertTrue(q.empty())
    
    def test_pop_2(self):
        q = Queue()
        q.put(2)
        q.put(1)
        self.assertEqual(q.pop(), 2)
        self.assertEqual(q.pop(), 1)
        self.assertTrue(q.empty())

    def test_pop_exception(self):
        with self.assertRaises(AssertionError):
            q = Queue()
            q.put(1)
            q.pop()
            q.pop()

if __name__ == '__main__':
    unittest.main()    

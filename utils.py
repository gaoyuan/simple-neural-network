class Queue:
    """A custom queue implementation."""
    class Node:
        """Node in a queue."""
        def __init__(self, x):
            self.val = x
            self.next = None
    
    def __init__(self):
        self.head = None
        self.tail = None
    def empty(self):
        return self.head is None
    def top(self):
        return self.head.val
    def put(self, x):
        if self.head is None:
            self.head = self.tail = self.Node(x)
        else:
            self.tail.next = self.Node(x)
            self.tail = self.tail.next
    def pop(self):
        assert self.head is not None
        value = self.top()
        self.head = self.head.next
        return value

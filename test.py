import unittest
from value import Value
import activation as act

class TestPygrad(unittest.TestCase):
    def test_value1(self):
        x, y = Value(2), Value(3)
        z = 4*x + y
        z.backward()

        self.assertAlmostEqual(x.data, 2)
        self.assertAlmostEqual(y.data, 3)
        self.assertAlmostEqual(z.data, 11)

        self.assertAlmostEqual(x.grad, 4)
        self.assertAlmostEqual(y.grad, 1)
        self.assertAlmostEqual(z.grad, 1)

        z.draw_graph('images/graph1')

    def test_activation1(self):
        x1 = Value(-1.0)
        x2 = Value(2.0)
        x3 = x1 + x2
        x4 = Value(3.0)
        x5 = x3 * x4
        x6 = act.tanh(x5)

        x6.backward()
        x6.draw_graph('images/graph2')

        self.assertAlmostEqual(x1.data, -1, delta=3)
        self.assertAlmostEqual(x2.data, 2, delta=3)
        self.assertAlmostEqual(x3.data, 1, delta=3)
        self.assertAlmostEqual(x4.data, 3, delta=3)
        self.assertAlmostEqual(x5.data, 3, delta=3)
        self.assertAlmostEqual(x6.data, 0.995, delta=3)

        self.assertAlmostEqual(x1.grad, 0.03, delta=3)
        self.assertAlmostEqual(x2.grad, 0.03, delta=3)
        self.assertAlmostEqual(x3.grad, 0.03, delta=3)
        self.assertAlmostEqual(x4.grad, 0.01, delta=3)
        self.assertAlmostEqual(x5.grad, 0.01, delta=3)
        self.assertAlmostEqual(x6.grad, 1, delta=3)

    def test_activation2(self):
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = act.relu(z) + z * x
        h = act.relu(z * z)
        y = h + q + q * x
        y.backward()
        
        self.assertAlmostEqual(x.data, -4, delta=3)
        self.assertAlmostEqual(z.data, -10, delta=3)
        self.assertAlmostEqual(q.data, 40, delta=3)
        self.assertAlmostEqual(h.data, 100, delta=3)
        self.assertAlmostEqual(y.data, -20, delta=3)
        
        self.assertAlmostEqual(x.grad, 46, delta=3)
      

if __name__ == '__main__':
    unittest.main()

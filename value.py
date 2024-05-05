import numpy as np
import graphviz

class Value():
    def __init__(self, data, op='', children=()):
        self.data = data
        self.backward_ = lambda: None
        self.grad = 0.0
        self.children = set(children)
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, '+', (self, other))
    
        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_ = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, '*', (self, other))
    
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward_ = backward
        return out

    def __pow__(self, n):
        out = Value(self.data**n, '^', (self,))

        def backward():
            self.grad += (n*self.data**(n-1)) * out.grad

        out.backward_ = backward
        return out

    def log(self):
        out = Value(np.log(self.data), 'log', (self,))
 
        def backward():
            self.grad += (1.0/self.data) * out.grad

        out.backward_ = backward
        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def backward(self):
        topo_sorted = []
        visited = set()

        def topo_sort(value):
            if value in visited:
                return
            visited.add(value)
            for child in value.children:
                topo_sort(child)
            topo_sorted.append(value)

        topo_sort(self)
        self.grad = 1.0
        for value in reversed(topo_sorted):
            value.backward_()

    def draw_graph(self, filename):
        graph = graphviz.Digraph(comment='computation graph', format='svg')
        visited = {}

        def draw(value):
            if value in visited:
                return visited[value]
            visited[value] = id(value)
            graph.node(str(id(value)), label=f'{id=} | data={value.data:.3f} | grad={value.grad:.3f}', shape='record')
            if value.op != '':
                graph.node(str(id(value))+value.op, value.op)
                graph.edge(str(id(value))+value.op, str(id(value)))
            for child in value.children:
                node_id = draw(child)
                graph.edge(str(node_id), str(id(value))+value.op)
            return id(value)

        draw(self)
        graph.render(filename)
            
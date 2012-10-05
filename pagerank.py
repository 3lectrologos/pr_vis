"""
PageRank simulation.
"""

__author__ = "Alkis Gkotovos"
__email__ = "alkisg@student.ethz.ch"


import numpy
import math
import networkx as nx
import matplotlib.pyplot as plt

class PRGraph(nx.DiGraph):
    INIT_NODE_SIZE = 500
    NODE_COLOR = '#5555EE'
    EDGE_COLOR = '0.2'
    EDGE_WIDTH = 2
    FONT_COLOR = '0.95'
    FONT_SIZE = 10

    def __init__(self, g, theta=0.85, pos=None):
        """Initialize with the given graph."""
        nx.DiGraph.__init__(self, g)
        self.pos = pos
        self.n = g.number_of_nodes()
        w = numpy.mat(self.n * [0.0]).transpose()
        for s in self.sinks():
            w[s] = 1
        ones = numpy.mat(self.n * [1.0]).transpose()
        self.H = nx.to_numpy_matrix(g) + w.dot(ones.transpose())
        for i in range(0, self.n):
            self.H[i] = self.H[i] / numpy.sum(self.H[i])
        ones_mat = (1.0 / self.n)*ones.dot(ones.transpose())
        self.G = theta * self.H + (1 - theta) * ones_mat
        self.reset()

    def sinks(self):
        non_sinks = set()
        for (u, v) in self.edges():
            non_sinks.add(self.nodes().index(u))
        return list(set([x - 1 for x in self.nodes()]) - non_sinks)

    def reset(self):
        self.ps = numpy.mat(self.n * [1.0 / self.n]).transpose()

    def plot(self, node_size=INIT_NODE_SIZE):
        if self.pos == None:
            self.pos = nx.graphviz_layout(self)
        plt.clf()
        nx.draw_networkx_nodes(self, pos=self.pos,
                               nodelist=self.nodes(),
                               node_color=self.NODE_COLOR,
                               node_size=self.get_node_sizes())
        nx.draw_networkx_edges(self, pos=self.pos,
                               edgelist=self.edges(),
                               width=self.EDGE_WIDTH,
                               edge_color=self.EDGE_COLOR)
        nx.draw_networkx_labels(self, pos=self.pos,
                                font_color=self.FONT_COLOR,
                                font_size=self.FONT_SIZE)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.draw()

    def get_node_sizes(self):
        a = 50.0
        return [max(min(a + (500.0 - a)*self.n*x,
                        numpy.mat(1000.0)),
                    numpy.mat(100.0)) for x in self.ps]

    def step(self):
        self.ps = self.G.transpose().dot(self.ps)
        self.ps = self.ps / numpy.sum(self.ps)

    def print_ps(self):
        print "Current importance vector = "
        print(self.ps)
        
def iterate(graph, theta=0.85):
    plt.ion()
    plt.figure()
    numpy.set_printoptions(precision=3)
    g = PRGraph(graph, theta=theta)
    g.plot()
    print "Press <Enter> to continue or 'q' followed by <Enter> to quit."
    g.print_ps()
    while True:
        inp = raw_input('')
        if inp.startswith('q'):
            break
        g.step()
        g.plot()
        g.print_ps()

def demo1():
    g = nx.DiGraph([(1, 2), (2, 1), (3, 1), (3, 3), (3, 5), (4, 3), (4, 5)])
    iterate(g)

def demo2():
    g = nx.scale_free_graph(30)
    iterate(g)

if __name__=="__main__":
    demo1()

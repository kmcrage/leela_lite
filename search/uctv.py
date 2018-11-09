import numpy as np
import math
import heapq
from collections import OrderedDict
import os


class UCTVNode():
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.total_vsquared = 0  # float
        self.number_visits = 0  # int

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def sigma(self):  # returns float
        return math.sqrt(self.total_vsquared / (1 + self.number_visits) - self.Q() ** 2)

    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits) / (1 + self.number_visits))

    def best_child(self, C, zeta):
        return max(self.children.values(),
                   key=lambda node: (node.Q() +
                                     C * node.prior * node.U() ** 2 +
                                     zeta * node.prior * node.U() * node.sigma()))

    def select_leaf(self, C, zeta):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C, zeta)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTVNode(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:            
            current.number_visits += 1
            current.total_value += (value_estimate * turnfactor)
            current.total_vsquared += value_estimate ** 2
            current = current.parent
            turnfactor *= -1
        current.number_visits += 1

    def dump(self, move, C):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q())
        print("U: ", self.U())
        print("BestMove: ", self.Q() + C * self.U())
        # print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")


def UCTV_search(board, num_reads, net=None, C=1.0, zeta=1.0):
    assert(net is not None)
    zeta = float(os.getenv('ZETA', zeta))
    C = float(os.getenv('C', C))
    root = UCTVNode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(C, zeta)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    size = min(5, len(root.children))
    pv = heapq.nlargest(size, root.children.items(),
                        key=lambda item: (item[1].number_visits, item[1].Q()))

    print('UCTV pv:', [(n[0], n[1].Q(), n[1].number_visits) for n in pv])
    return max(root.children.items(),
               key=lambda item: (item[1].number_visits, item[1].Q()))



#num_reads = 10000
#import time
#tick = time.time()
#UCT_search(GameState(), num_reads)
#tock = time.time()
#print("Took %s sec to run %s times" % (tock - tick, num_reads))
#import resource
#print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

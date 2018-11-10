import numpy as np
import math
import heapq
from collections import OrderedDict
import os

"""
Asymmetric Move Selection Strategies in
Monte-Carlo Tree Search: Minimizing the Simple
Regret at Max Nodes
"""


class SRCRNode:
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[SRCRNode]
        self.children = OrderedDict()  # Dict[move, SRCRNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U_sr(self):  # returns float
        """
        this is the classic simple regret minimiser, used for max (self) nodes
        :return:
        """
        return self.prior * math.sqrt(math.sqrt(self.parent.number_visits) / (1 + self.number_visits))

    def U_cr(self):  # returns float
        """
        this is the classic cumulative regret minimiser, used for min (opponent) nodes
        :return:
        """
        return self.prior * math.sqrt(math.log(self.parent.number_visits) / (1 + self.number_visits))

    def best_child(self, C_sr, C_cr):
        return max(self.children.values(),
                   key=lambda node: node.Q() + C_sr * node.U_sr() + C_cr * node.U_cr())

    def select_leaf(self, C_sr=1.0, C_cr=1.0):
        current = self
        depth = 0
        while current.is_expanded and current.children:
            if depth % 2 == 0:
                current = current.best_child(C_sr, 0.)  # MAX node, simple regret
            else:
                current = current.best_child(0., C_cr)  # MIN node, cumulative regret
            depth += 1
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = SRCRNode(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:            
            current.number_visits += 1
            current.total_value += (value_estimate *
                                    turnfactor)
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
        print("U_cr: ", self.U_cr())
        print("U_sr: ", self.U_sr())
        # print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")


def SRCR_search(board, num_reads, net=None, C_sr=3.4, C_cr=3.4):
    assert(net is not None)
    C_sr = float(os.getenv('CP_SR', C_sr))
    C_cr = float(os.getenv('CP_CR', C_cr))
    root = SRCRNode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(C_sr, C_cr)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    size = min(5, len(root.children))
    pv = heapq.nlargest(size, root.children.items(),
                        key=lambda item: (item[1].number_visits, item[1].Q()))

    print('SRCR (', C_sr, ',', C_cr, ') pv:', [(n[0], n[1].Q(), n[1].number_visits,
                                          C_sr*n[1].U_sr(), C_cr*n[1].U_cr()) for n in pv])
    print('prediction:', end=' ')
    next = pv[0]
    while len(next[1].children):
        next = heapq.nlargest(1, next[1].children.items(),
                                key=lambda item: (item[1].number_visits, item[1].Q(alpha)))[0]
        print(next[0], end=' ')
    print('')
    return max(root.children.items(),
               key=lambda item: (item[1].number_visits, item[1].Q()))

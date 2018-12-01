import heapq
from collections import OrderedDict
import numpy
import math
import random
import weakref

"""
Normal Sampling
initialise Q from parent
"""


class NSNode:
    name = 'ns'

    def __init__(self, board=None, parent=None, move=None, prior=0, sse=0.1, beta=0.5, verbose=True):
        self.board = board
        self.move = move
        self.is_expanded = False
        self._parent = weakref.ref(parent) if parent else None  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.beta = beta
        self.prior = prior  # float
        self.q = -parent.q if parent else 0  # float, fpu is in lc0 rather than a0
        self.q_sse = sse
        self.number_visits = 1

        self.verbose = verbose

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def sigma(self):
        return math.sqrt(self.q_sse / self.number_visits)

    def confidence(self):
        return self.prior * math.sqrt(self.parent.number_visits / self.number_visits)

    def best_child(self):
        """
        pick the top result beta of the time, else the second top
        pick using the optimistic sampler
        :return:
        """
        num = min(2, len(self.children))
        candidates = heapq.nlargest(num,
                                    self.children.values(),
                                    key=lambda node: max(node.q,
                                                         numpy.random.normal(node.q,
                                                                             node.confidence() * node.sigma())))
        if num == 1 or random.random() < self.beta:
            return candidates[0]
        return candidates[1]

    def select_leaf(self):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child()
        return current

    def expand(self, child_priors):
        if self.is_expanded:
            return
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        board = self.board.copy()
        board.push_uci(move)
        self.children[move] = self.__class__(parent=self, move=move, prior=prior, board=board)
    
    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        sample = -value_estimate
        while current:
            current.number_visits += 1
            e = sample - current.q
            current.q += e / current.number_visits
            current.q_sse += e * (sample - current.q)
            current = current.parent
            sample *= -1

    def get_node(self, move):
        if move in self.children:
            return self.children[move]
        return None

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda item: (item[1].number_visits, item[1].q))
        if self.verbose:
            print(self.name, 'pv:', [(n[0], n[1].q, n[1].sigma(), n[1].number_visits) for n in pv])
        # there could be no moves if we jump into a mate somehow
        print('prediction:', end=' ')
        predict = pv[0]
        while len(predict[1].children):
            predict = heapq.nlargest(1, predict[1].children.items(),
                                     key=lambda item: (item[1].number_visits, item[1].q))[0]
            print(predict[0], end=' ')
        print('')
        return pv[0] if pv else None


class NSMinusNode(NSNode):
    name = 'ns_minus'

    def __init__(self, **kwargs):
        super().__init__(beta=0.25, **kwargs)

class NSPlusNode(NSNode):
    name = 'ns_plus'

    def __init__(self, **kwargs):
        super().__init__(beta=0.0, **kwargs)

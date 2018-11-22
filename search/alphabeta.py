import math
import heapq
from collections import OrderedDict

"""
A Rollout-Based Search Algorithm Unifying MCTS and Alpha-Beta
"""


class ABNode:
    name = 'ab'

    def __init__(self, board=None, parent=None, move=None, prior=0,
                 k=9, verbose=True):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = []
        self.prior = prior  # float

        self.eval = 0
        self.alpha = -1
        self.beta = 1
        self.depth = 0
        self.v_minus = [-1] * 10 # max depth
        self.v_plus = [1] * 10
        self.best_child = None

        self.k = k
        self.verbose = verbose

    def select_leaf(self):
        d = self.depth
        alpha = self.alpha
        beta = self.beta
        current = self
        while current.is_expanded and current.children and d:
            feasible_children = []
            for child in current.children:
                child.alpha = max(alpha, -child.v_plus[d-1])
                child.beta = min(beta, -child.v_minus[d-1])
                if child.alpha < child.beta:
                    feasible_children.append(child)
            current = feasible_children[0]
            d -= 1
            alpha = -current.beta
            beta = -current.alpha
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current, d

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items()[:self.k]:
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children.append(self.__class__(parent=self, move=move, prior=prior))
    
    def backup(self, value_estimate, d):
        current = self
        current.eval = value_estimate
        while current.parent is not None:
            current = current.parent
            d += 1
            current.v_minus[d] = -min([c.v_plus[d-1] for c in current.children])
            current.v_plus[d] = -min([c.v_minus[d-1] for c in current.children])
        if current.v_minus[current.d] == current.v_plus[current.d]:
            current.best_child = self.get_best_child(d)
            current.d += 1

    def get_best_child(self, d):
        best_child = self
        v = self.v_plus[d]
        while d:
            d -= 1
            best_child =[c for c in best_child.children if c.v_plus[d] == v]
            print('best child', d, best_child.move)
        return best_child

    def get_node(self, move):
        if move in self.children:
            return self.children[move]
        return None

    def dump(self):
        print("---")
        print("move: ", self.move)
        print("prior: ", self.prior)
        print("---")

    def outcome(self):
        return self.best_child.move, self.best_child

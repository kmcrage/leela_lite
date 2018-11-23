import math
import heapq
from collections import defaultdict

"""
A Rollout-Based Search Algorithm Unifying MCTS and Alpha-Beta
"""

# this is used in float comparisons
TOLERANCE = 1e-8
LOSS = -1
WIN = 1


class ABNode:
    name = 'ab'

    def __init__(self, board=None, parent=None, move=None, prior=0,
                 k=5, verbose=True):
        # game state
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = []
        self.prior = prior  # float

        # search parms
        self.k = k
        self.verbose = verbose

        # eval
        self.eval = 0
        self.alpha = LOSS
        self.beta = WIN
        self.depth = 0
        self.v_minus = defaultdict(lambda: LOSS)
        self.v_plus = defaultdict(lambda: WIN)
        self.best_child = None

        # rewards
        self.weight = 10
        self.wscale = k + 1
        self.number_visits = 0

    def select_leaf(self):
        d = self.depth
        alpha = self.alpha
        beta = self.beta
        current = self
        # print('selct leaf', d, alpha, beta)
        while current.is_expanded and current.children and d:
            feasible_children = []
            for child in current.children:
                child.alpha = max(alpha, child.v_minus[d-1])
                child.beta = min(beta, child.v_plus[d-1])
                # print('child', child.move, child.v_minus[d-1], child.v_plus[d-1], child.alpha, child.beta)
                if child.alpha < child.beta:
                    feasible_children.append(child)
            current = feasible_children[0]
            d -= 1
            alpha = -current.beta
            beta = -current.alpha
            # print('selcting', [c.move for c in feasible_children])
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        """
        expand best k children, ordered by priors
        :param child_priors:
        :return:
        """
        self.is_expanded = True
        # print('child priors', child_priors)
        for move, prior in (list(child_priors.items()))[:self.k]:
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children.append(self.__class__(parent=self, move=move, prior=prior))
    
    def backup(self, value_estimate):
        """
        minmax backup of alpha, beta values, increase depth counter if minmax solves that depth
        :param value_estimate:
        :return:
        """
        current = self
        d = 0
        current.v_plus[d] = value_estimate
        current.v_minus[d] = value_estimate
        # print('backup', current.move, value_estimate)
        while current.parent is not None:
            current = current.parent
            d += 1
            current.v_minus[d] = -min([c.v_plus[d-1] for c in current.children])
            current.v_plus[d] = -min([c.v_minus[d-1] for c in current.children])
            # print('est', current.depth, current.move, current.v_minus[current.depth], current.v_plus[current.depth])

    def update_root(self):
        if math.fabs(self.v_minus[self.depth] - self.v_plus[self.depth]) > TOLERANCE:
            return
        for c in self.children:
            # print('vplus', c.move, d, c.v_plus, c.v_minus, self.v_plus)
            if math.fabs(self.v_plus[self.depth] + c.v_plus[self.depth-1]) < TOLERANCE:
                c.number_visits = self.weight * math.pow(self.wscale, self.depth)
        self.depth += 1
        print('new depth', self.depth)

    def get_node(self, move):
        for node in self.children:
            if node.move == move:
                return node
        return None

    def dump(self):
        print("---")
        print("move: ", self.move)
        print("prior: ", self.prior)
        print("---")

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children,
                            key=lambda item: (item.number_visits, item.prior))
        if self.verbose:
            print(self.name, 'pv:', [(n.move, n.v_plus[self.depth - 2], n.number_visits, n.prior) for n in pv])
        return pv[0].move, pv[0]

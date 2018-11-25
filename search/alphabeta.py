import math
import heapq
from collections import defaultdict
import weakref

"""
A Rollout-Based Alpha-Beta Search Algorithm
"""

# this is used in float comparisons
TOLERANCE = 1e-8
LOSS = -1
WIN = 1


class ABNode:
    name = 'ab'

    def __init__(self, board=None, parent=None, move=None, prior=0,
                 k=5, verbose=True):
        self.verbose = verbose
        # game state
        self.board = board
        self.move = move
        self.is_expanded = False
        self._parent = weakref.ref(parent) if parent else None  # Optional[UCTNode]
        self.children = []
        self.prior = prior  # float

        # search parms
        self.k = k

        # eval
        self.depth = 0
        self.v_minus = defaultdict(lambda: LOSS)
        self.v_plus = defaultdict(lambda: WIN)

        # rewards
        self.weight = 1
        self.wscale = k
        self.bonus_visits = 0

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def select_leaf(self):
        while math.fabs(self.v_minus[self.depth] - self.v_plus[self.depth]) < TOLERANCE:
            self.update_root()  # this also increments depth

        alpha = -self.v_plus[self.depth]
        beta = -self.v_minus[self.depth]
        d = self.depth
        current = self

        # print('selct leaf', d, alpha, beta)
        while current.is_expanded and current.children and d:
            feasible_children = []
            for child in current.children:
                child_alpha = max(alpha, child.v_minus[d-1])
                child_beta = min(beta, child.v_plus[d-1])
                # print('child', d, child.move, child.v_minus[d-1], child.v_plus[d-1], child_alpha, child_beta)
                if child_alpha < child_beta:
                    feasible_children.append(child)
            # it is proven by Huang that this set is never empty
            current = feasible_children[0]
            d -= 1
            alpha, beta = -min(beta, current.v_plus[d]), -max(alpha, current.v_minus[d])
            # print('selcting', [c.move for c in feasible_children])

        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)

        return current

    def update_root(self):
        d = self.depth
        nxt = self
        bonus = self.weight * math.pow(self.wscale, self.depth)
        while d:
            candidates = [c for c in nxt.children if math.fabs(nxt.v_plus[d] + c.v_minus[d-1]) < TOLERANCE]
            if not candidates:
                break
            nxt = candidates[0]
            nxt.bonus_visits = bonus
            d -= 1
        self.depth += 1

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
        minmax backup of alpha, beta values
        values are stored as value to opponent reward rather than self value
        :param value_estimate:
        :return:
        """
        current = self
        d = 0
        current.v_plus[d] = -value_estimate
        current.v_minus[d] = -value_estimate
        # print('backup', current.move, value_estimate)
        while current.parent is not None:
            current = current.parent
            d += 1
            current.v_minus[d] = -max([c.v_plus[d-1] for c in current.children])
            current.v_plus[d] = -max([c.v_minus[d-1] for c in current.children])
            # print('est', current.depth, current.move, current.v_minus[current.depth], current.v_plus[current.depth])

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
                            key=lambda item: (item.bonus_visits, item.prior))
        if self.verbose:
            print(self.name, 'pv:', [(n.move, n.v_plus[self.depth - 2], n.bonus_visits, n.prior) for n in pv])

            d = self.depth - 1
            nxt = pv[0]
            print('prediction:', end=' ')
            while d:
                candidates = [c for c in nxt.children if math.fabs(nxt.v_plus[d] + c.v_minus[d - 1]) < TOLERANCE]
                if not candidates:
                    break
                nxt = candidates[0]
                print(nxt.move, end=' ')
                d -= 1
            print('')
        return pv[0].move, pv[0]

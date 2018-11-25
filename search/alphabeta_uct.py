import math
import heapq
from collections import defaultdict
import random
import weakref

"""
A Rollout-Based Alpha-Beta Search Algorithm
"""

# this is used in float comparisons
TOLERANCE = 1e-8
LOSS = -1
WIN = 1


class ABUCTNode:
    name = 'abuct'

    def __init__(self, board=None, parent=None, move=None, prior=0,
                 k=5, cpuct=3.4, p=0., weight=1, wscale=5, verbose=True):
        self.verbose = verbose
        # game state
        self.board = board
        self.move = move
        self.is_expanded = False
        self._parent = weakref.ref(parent) if parent else None  # Optional[UCTNode]
        self.children = []
        self.prior = prior  # float

        # search parms
        self.k = k  # pruning for ab
        self.cpuct = cpuct  # width for uct
        self.p = p  # ab:0 vs uct:1

        # eval
        self.depth = 0
        self.v_minus = defaultdict(lambda: LOSS)
        self.v_plus = defaultdict(lambda: WIN)
        self.total_value = -parent.Q() if parent else 0
        self.bonus_visits = 0
        self.number_visits = 0

        # ab reward conversion
        self.weight = weight
        self.wscale = wscale

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        sibling_visits = sum([c.number_visits + c.bonus_visits for c in self.parent.children])
        return math.sqrt(sibling_visits) * self.prior / (1 + self.number_visits + self.bonus_visits)

    def best_child_uct(self):
        return max(self.children,
                   key=lambda node: node.Q() + self.cpuct * node.U())

    def select_leaf(self):
        current = self
        while current.is_expanded and current.children:
            if random.random() < self.p:
                current = current.best_child_uct()
            else:
                current = self.select_leaf_ab()
                break
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def ab_children(self):
        ab_children = [c for c in self.children if c.number_visits > 0]
        if len(ab_children) < self.k:
            ab_children += heapq.nlargest(self.k - len(ab_children),
                                          [c for c in self.children if not c.number_visits],
                                          key=lambda c: c.prior)
        return ab_children

    def select_leaf_ab(self):
        print(self.depth, self.v_minus[self.depth], self.v_plus[self.depth])
        while math.fabs(self.v_minus[self.depth] - self.v_plus[self.depth]) < TOLERANCE:
            self.update_root()  # this also increments depth
            print(self.depth, self.v_minus[self.depth], self.v_plus[self.depth])

        alpha = -self.v_plus[self.depth]
        beta = -self.v_minus[self.depth]
        d = self.depth
        current = self

        print('selct leaf', d, alpha, beta)
        while current.is_expanded and current.ab_children() and d:
            feasible_children = []
            for child in current.ab_children():
                child_alpha = max(alpha, child.v_minus[d-1])
                child_beta = min(beta, child.v_plus[d-1])
                #print('child', d, child.move, child.v_minus[d-1], child.v_plus[d-1], child_alpha, child_beta)
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
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children.append(self.__class__(parent=self, move=move, prior=prior))
    
    def backup(self, value_estimate):
        """
        minmax backup of alpha, beta values
        averaging backup of uxt value
        values are stored as value to opponent reward rather than self value
        :param value_estimate:
        :return:
        """
        current = self
        d = 0
        current.v_plus[d] = -value_estimate
        current.v_minus[d] = -value_estimate
        current.total_value += -value_estimate
        current.number_visits += 1
        turnfactor = -1
        # print('backup', current.move, value_estimate)
        while current.parent is not None:
            current = current.parent
            d += 1
            current.v_minus[d] = -max([c.v_plus[d-1] for c in current.children])
            current.v_plus[d] = -max([c.v_minus[d-1] for c in current.children])
            # print('est', current.depth, current.move, current.v_minus[current.depth], current.v_plus[current.depth])
            current.number_visits += 1
            turnfactor *= -1
            current.total_value += (value_estimate * turnfactor)

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
                            key=lambda item: (item.number_visits + item.bonus_visits, item.Q(), item.prior))
        if self.verbose:
            print(self.name, 'pv:', [(n.move, n.Q(), n.v_plus[n.depth-1], n.number_visits, n.bonus_visits) for n in pv])

            print('prediction:', end=' ')
            predict = pv[0]
            while len(predict.children):
                predict = heapq.nlargest(1, predict.children,
                                         key=lambda item: (item.number_visits + item.bonus_visits, item.Q()))[0]
                print(predict.move, end=' ')
            print('')
        return pv[0].move, pv[0]

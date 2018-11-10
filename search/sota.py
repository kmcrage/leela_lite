import math
import heapq
from collections import OrderedDict
# import os

"""
Asymmetric Move Selection Strategies in
Monte-Carlo Tree Search: Minimizing the Simple
Regret at Max Nodes
"""


class SOTANode:
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[SOTANode]
        self.children = OrderedDict()  # Dict[move, SOTANode]
        self.prior = prior  # float

        self.reward = 0.  # float
        self.minmax_value = 0.  # float
        self.bellman_value = 0.  # float

        self.number_visits = 0  # int
        self.leaf_visits = 0  # int

    def Q(self, alpha):  # returns float
        return (1 - alpha) * self.bellman_value + alpha * self.minmax_value

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

    def best_child(self, C_sr, C_cr, alpha):
        def node_order(node):
            return (node.Q(alpha) + C_sr * node.U_sr() + C_cr * node.U_cr(),
                    node.prior)
        return max(self.children.values(), key=node_order)

    def select_leaf(self, C_max_sr, C_max_cr, C_min_sr, C_min_cr, alpha):
        current = self
        depth = 0
        while current.is_expanded and current.children:
            if depth % 2 == 0:
                current = current.best_child(C_max_sr, C_max_cr, alpha)  # MAX node, simple regret
            else:
                current = current.best_child(C_min_sr, C_min_cr, alpha)  # MIN node, cumulative regret
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
        self.children[move] = SOTANode(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        self.reward = -value_estimate
        self.bellman_value = self.reward
        self.minmax_value = self.reward
        self.leaf_visits += 1
        self.number_visits += 1

        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            current.minmax_value = -max([n.minmax_value for n in current.children.values()
                                         if n.number_visits])
            # do we want to add in this reward? its more stable and will disappear with many evals
            # keep because it matters on leaf nodes
            current.bellman_value = current.reward * current.leaf_visits / current.number_visits
            for child in [n for n in current.children.values() if n.number_visits]:
                current.bellman_value -= child.number_visits * child.bellman_value / current.number_visits
                # print('updating Q:', current.Q, current.number_visits, visits)
            # print('postupdate Q:', current.Q, current.number_visits, visits)

        current.number_visits += 1

    def dump(self, move):
        print("---")
        print("move: ", move)
        print("belman value: ", self.bellman_value)
        print("minmax value: ", self.minmax_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("U_cr: ", self.U_cr())
        print("U_sr: ", self.U_sr())
        # print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")


def SOTA_search(board, num_reads, net=None,
                C_max_sr=3.4, C_max_cr=0.,
                C_min_sr=0., C_min_cr=3.4,
                alpha=0.):
    assert(net is not None)
    # C_sr = float(os.getenv('CP_SR', C_sr))
    # C_cr = float(os.getenv('CP_CR', C_cr))
    root = SOTANode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(C_max_sr, C_max_cr, C_min_sr, C_min_cr, alpha)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    size = min(5, len(root.children))
    pv = heapq.nlargest(size, root.children.items(),
                        key=lambda item: (item[1].number_visits, item[1].Q(alpha)))
    #
    print('SOTA pv:', [(n[0], n[1].Q(alpha), n[1].number_visits) for n in pv])
    # print('prediction:', end=' ')
    # next = pv[0]
    # while len(next[1].children):
    #    next = heapq.nlargest(1, next[1].children.items(),
    #                            key=lambda item: (item[1].number_visits, item[1].Q(alpha)))[0]
    #    print(next[0], end=' ')
    # print('')
    return pv[0]

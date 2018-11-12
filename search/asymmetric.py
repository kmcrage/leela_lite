import math
from search.uct import UCTNode

"""
Asymmetric Move Selection Strategies in
Monte-Carlo Tree Search: Minimizing the Simple
Regret at Max Nodes
"""


class AsymNode(UCTNode):
    name = 'asym'

    def __init__(self, C_max_sr=3.4, C_max_cr=0., C_min_sr=0., C_min_cr=3.4, **kwargs):
        super(AsymNode, self).__init__(**kwargs)
        self.C_max_sr = C_max_sr
        self.C_max_cr = C_max_cr
        self.C_min_sr = C_min_sr
        self.C_min_cr = C_min_cr

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
        return self.prior * math.sqrt(math.log(1 + self.parent.number_visits) / (1 + self.number_visits))

    def best_child(self, C_sr, C_cr):
        return max(self.children.values(),
                   key=lambda node: node.Q() + C_sr * node.U_sr() + C_cr * node.U_cr())

    def select_leaf(self):
        current = self
        depth = 0
        while current.is_expanded and current.children:
            if depth % 2 == 0:
                current = current.best_child(self.C_max_sr, self.C_max_cr)  # MAX node, simple regret
            else:
                current = current.best_child(self.C_min_sr, self.C_min_cr)  # MIN node, cumulative regret
            depth += 1
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

import math
import heapq
from collections import OrderedDict

"""
Standard UCT
"""


class UCTNode:
    def __init__(self, board=None, parent=None, move=None, prior=0,
                 cpuct=3.4):
        self.cpuct = cpuct

        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = -parent.Q() if parent else 0.  # float
        self.number_visits = 0  # int

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)

    def best_child(self):
        return max(self.children.values(),
                   key=lambda node: node.Q() + self.cpuct * node.U())

    def select_leaf(self):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child()
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(parent=self, move=move, prior=prior)
    
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

    def dump(self):
        print("---")
        print("move: ", self.move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q())
        print("U: ", self.U())
        print("BestMove: ", self.Q() + self.cpuct * self.U())
        # print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda item: (item[1].number_visits, item[1].Q()))

        print(self.__class__, 'pv:', [(n[0], n[1].Q(), n[1].U(), n[1].number_visits) for n in pv])
        return pv[0]
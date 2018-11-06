import math
import heapq
from collections import OrderedDict


class MinMaxNode:
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[MinMaxNode]
        self.children = OrderedDict()  # Dict[move, MinMaxNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int
        self.Q = 0

    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits)
                * self.prior / (1 + self.number_visits))

    def best_child(self, C):
        return max(self.children.values(),
                   key=lambda node: node.Q + C*node.U())

    def select_leaf(self, C):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = MinMaxNode(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        current.Q = -value_estimate
        current.number_visits += 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            current.Q = -max([n.Q for n in current.children.values() if n.number_visits])
        current.number_visits += 1

    def dump(self, move, C):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q)
        print("U: ", self.U())
        print("BestMove: ", self.Q + C * self.U())
        # print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")


def MinMax_search(board, num_reads, net=None, C=1.0):
    assert(net is not None)
    root = MinMaxNode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(C)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    size = min(5, len(root.children))
    pv = heapq.nlargest(size, root.children.items(),
                        key=lambda item: (item[1].number_visits, item[1].Q()))

    print('MinMax pv:', [(n[0], n[1].Q(), n[1].number_visits) for n in pv])
    return max(root.children.items(),
               key=lambda item: (item[1].number_visits, item[1].Q()))

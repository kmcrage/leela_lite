import math
import heapq
from collections import OrderedDict

"""
Standard UCT
"""


class UCTNode:
    name = 'uct'

    def __init__(self, board=None, parent=None, move=None, prior=0,
                 cpuct=3.4):
        self.cpuct = cpuct

        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = 0.  # float
        self.number_visits = 0  # int
        self.reward = 0

    def Q(self):  # returns float
        """
        Q(action, state) action value estimate: the value of moving to this state (parent pov)
        :return:
        """
        return self.reward + self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        """
        an upper confidence bound on the value of the node, from our parents pov
        :return:
        """
        return math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)

    def V(self):
        """
        V(state) state value estimate: the value of the state (self pov)
        :return:
        """
        children = [n for n in self.children.values() if n.number_visits]
        if children:
            value = max([n.Q() for n in children])
        else:
            value = -self.Q()
        return value

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
        self.children[move] = self.__class__(parent=self, move=move, prior=prior)
    
    def backup(self, value_estimate: float):
        current = self
        current.reward = -value_estimate
        current.total_value = current.reward
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.total_value += (value_estimate * turnfactor)
            current.reward = 0.

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

        print(self.name, 'pv:', [(n[0], n[1].Q(), n[1].U(), n[1].number_visits) for n in pv])

        next = pv[0]
        print('prediction:', next[0], end=' ')
        while len(next[1].children):
            next = heapq.nlargest(1, next[1].children.items(),
                                    key=lambda item: (item[1].number_visits, item[1].Q()))[0]
            print(next[0], end=' ')
        print('')

        return pv[0]


class adaptive_mixin:
    def __init__(self, **kwargs):
        super(adaptive_mixin, self).__init__(**kwargs)

    def best_child(self):
        return max(self.children.values(),
                   key=lambda node: node.Q() + self.cpuct * (1 + node.V()) * node.U())


class AdaptNode(adaptive_mixin, UCTNode):
    name = 'adapt'

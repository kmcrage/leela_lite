"""
    value of information search

    MCTS Based on Simple Regret
    David Tolpin, Solomon Eyal Shimony
    https://pdfs.semanticscholar.org/2a81/bfc05ddec612fd9bf0aafad0a86ad13b0361.pdf

    Selecting Computations:  Theory and Applications
    https://arxiv.org/pdf/1207.5879.pdf
"""
import math
import heapq
from collections import OrderedDict


class VOINode:
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = -parent.Q() if parent else 0  # float
        self.reward = 0
        self.number_visits = 0  # int
        self.q_visits = 0  # int
        self.V = 0

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)

    def desirability(self, c):
        return self.Q() + c * self.U()

    def best_child_uct(self, c):
        return max(self.children.values(),
                   key=lambda node: node.desirability(c))

    def best_child_voi(self, c):
        """
        Take care here: bear in mind that the rewards in the papers are in the region [0, 1].
        Note that the formula in the  two papers are not identical.

        :return: best child
        """
        if len(self.children) < 2:
            return (list(self.children.values()))[0]

        # get the two best nodes by value, tie-break with visits, then prior for the special case of first move
        alpha, beta = heapq.nlargest(2,
                                     self.children.values(),
                                     key=lambda node: (node.Q(), node.q_visits, node.prior)
                                     )

        # this incorporates a /4 scaling as our reward has a range of 2, and we are squaring it
        phi = 2 * (math.sqrt(2) - 1) ** 2
        result = None
        vmax = -1
        for n in self.children.values():
            voi = n.prior / (1. + n.number_visits)
            if n == alpha:
                voi *= (1 + beta.Q()) * math.exp(-phi * alpha.number_visits * (alpha.Q() - beta.Q()) ** 2)
            else:
                voi *= (1 - alpha.Q()) * math.exp(-phi * n.number_visits * (alpha.Q() - n.Q()) ** 2)
            n.V = voi + n.Q()
            if voi > vmax:
                vmax = voi
                result = n
        self.V = vmax
        return result

    def select_leaf(self, c):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child_voi(c)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = VOINode(parent=self, move=move, prior=prior)

    def backup(self, value_estimate: float, c):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        current.number_visits += 1
        current.reward = -value_estimate
        current.total_value += current.reward
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            policy_move = current.best_child_uct(c)
            policy_move.q_visits += 1
            current.total_value -= policy_move.Q()
        # policy move at root
        policy_move = current.best_child_uct(c)
        policy_move.q_visits += 1
        current.number_visits += 1

    def dump(self, move):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q)
        print("---")

    def get_node(self, move):
        if move in self.children:
            return self.children[move]
        return None


def VOI_search(board, num_reads, net=None, c=3.4, root=None):
    assert(net is not None)
    if not root:
        root = VOINode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(c)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate, c)

    # the best node might be in second place because we've been trying to catch up with first place
    # so get the two best nodes and then check the value
    pv = sorted([i for i in root.children.items() if i[1].number_visits],
                key=lambda item: (item[1].Q(), item[1].q_visits), reverse=True)
    print('VOI pv:', [(n[0], n[1].Q(), n[1].number_visits, n[1].q_visits) for n in pv])
    print('VOI:', root.V)
    return pv[0]

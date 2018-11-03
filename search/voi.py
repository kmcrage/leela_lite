"""
    value of information search

    MCTS Based on Simple Regret
    David Tolpin, Solomon Eyal Shimony
    https://pdfs.semanticscholar.org/2a81/bfc05ddec612fd9bf0aafad0a86ad13b0361.pdf

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
        self.total_value = 0  # float
        self.number_visits = 0  # int
        self.V = 0

    @property
    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    @property
    def U(self):  # returns float
        return math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)

    def best_child_uct(self, c):
        return max(self.children.values(),
                   key=lambda node: node.Q + c*node.U)

    def best_child_voi(self):
        """
        Take care here: bear in mind that the rewards in the paper are in the region [0, 1].
        :param c: magic constant balancing information gain with reward value
        :return: best child
        """
        if len(self.children) < 2:
            return (list(self.children.values()))[0]

        # get the two best nodes by value, tie-break with visits, then prior for the special case of first move
        alpha, beta = heapq.nlargest(2,
                                     self.children.values(),
                                     key=lambda node: (node.Q, node.number_visits, node.prior)
                                     )
        # always try the best move, we don't want to choose something never evalled
        # if not alpha.number_visits:
        #    return alpha

        phi = 2 * (math.sqrt(2) - 1) ** 2
        result = None
        max = -1
        for n in self.children.values():
            voi = n.prior / (1. + n.number_visits)
            # the 0.5 here comes from q having a range of 2
            if n == alpha:
                voi *= (1 + beta.Q) * math.exp(-phi * alpha.number_visits * (alpha.Q - beta.Q) ** 2)
            else:
                voi *= (1 - alpha.Q) * math.exp(-phi * n.number_visits * (alpha.Q - n.Q) ** 2)
            n.V = voi
            if voi > max:
                max = voi
                result = n

        return result

    def select_leaf(self, c):
        current = self
        depth = 0
        while current.is_expanded and current.children:
            if depth:
                current = current.best_child_uct(c)
            else:
                current = current.best_child_voi()
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

    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate * turnfactor
            current = current.parent
            turnfactor *= -1
        # this is the root
        current.number_visits += 1

    def dump(self, move):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q)
        print("---")


def VOI_search(board, num_reads, net=None, c=1.0):
    assert(net is not None)
    root = VOINode(board)
    for _ in range(num_reads):
        leaf = root.select_leaf(c)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    pv = sorted(root.children.items(), key=lambda item: (item[1].number_visits, item[1].Q), reverse=True)

    print('VOI pv:', [(n[0], n[1].Q, n[1].number_visits, n[1].V) for n in pv])
    return pv[0]
    #return max(root.children.items(),
    #           key=lambda item: (item[1].Q, item[1].number_visits))

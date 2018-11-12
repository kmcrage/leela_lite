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
from search.uct import UCTNode


class VOINode(UCTNode):
    def __init__(self, **kwargs):
        super(VOINode, self).__init__(**kwargs)

    def best_child(self):
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
                                     key=lambda node: (node.Q(), node.number_visits, node.prior)
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
            if voi > vmax:
                vmax = voi
                result = n
        return result

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda item: (item[1].number_visits, item[1].Q()))
        best = pv[1] if len(pv) > 1 and pv[1][1].Q() > pv[0][1].Q() else pv[0]

        print(self.name, 'pv:', [(n[0], n[1].Q(), n[1].U(), n[1].number_visits) for n in pv])

        prediction = best
        print('prediction:', prediction[0], end=' ')
        while len(prediction[1].children):
            prediction = heapq.nlargest(1, prediction[1].children.items(),
                                        key=lambda item: (item[1].number_visits, item[1].Q()))[0]
            print(prediction[0], end=' ')
        print('')

        return best

import math
from search.uct import UCTNode
import os
import numpy
import heapq


class Thompson_mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prior_weight = float(os.getenv("PW", "5.0"))
        self.result_weight = float(os.getenv("RW", "1.0"))
        self.num_wins = 1 + self.prior_weight * self.prior  # from pov of parent
        self.num_losses = 1 + self.prior_weight * (1 - self. prior)

    def best_child(self):
        return max(self.children.values(),
                   key=lambda node: numpy.random.beta(node.num_wins, node.num_losses))

    def backup(self, value_estimate: float):
        current = self
        turnfactor = -1
        current.num_wins += self.result_weight * (1 - value_estimate * turnfactor)
        current.num_losses += self.result_weight * (1 + value_estimate * turnfactor)
        current.number_visits += 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.num_wins += self.result_weight * (1 - value_estimate * turnfactor)
            current.num_losses += self.result_weight * (1 + value_estimate * turnfactor)

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda item: (item[1].number_visits, item[1].Q()))
        if self.verbose:
            print(self.name, 'pv:', [(n[0],
                                      (n[1].num_wins - n[1].num_losses) /(n[1].num_wins+n[1].num_losses),
                                      n[1].number_visits) for n in pv])
        # there could be no moves if we jump into a mate somehow
        print('prediction:', end=' ')
        predict = pv[0]
        while len(predict[1].children):
            predict = heapq.nlargest(1, predict[1].children.items(),
                                     key=lambda item: (item[1].number_visits, item[1].Q()))[0]
            print(predict[0], end=' ')
        print('')
        return pv[0] if pv else None


class UCTTNode(Thompson_mixin, UCTNode):
    name = 'uctt'


import math
from search.uct import UCTNode
import os
import numpy


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
        current.num_wins += self.result_weight * (1 - value_estimate)
        current.num_losses += self.result_weight * (1 + value_estimate)
        current.number_visits += 1
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.num_wins += self.result_weight * (1 - value_estimate * turnfactor)
            current.num_losses += self.result_weight * (1 + value_estimate * turnfactor)


class UCTTNode(Thompson_mixin, UCTNode):
    name = 'uctt'


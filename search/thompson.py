import math
from search.uct import UCTNode
import os
import numpy


class Thompson_mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_wins = 20 * self.prior  # from pov of parent
        self.num_losses = 20 * (1 - self. prior)

    def best_child(self):
        return max(self.children.values(),
                   key=lambda node: numpy.random.beta(node.num_losses, node.num_wins))

    def backup(self, value_estimate: float):
        current = self
        current.num_wins += 1 - value_estimate
        current.num_losses += 1 + value_estimate
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.num_wins += 1 - value_estimate * turnfactor
            current.num_losses += 1 + value_estimate * turnfactor


class UCTTNode(Thompson_mixin, UCTNode):
    name = 'uctt'


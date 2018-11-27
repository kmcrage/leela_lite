import math
from search.uct import UCTNode
import os
import numpy
import heapq


class Thompson_mixin:
    def __init__(self, action_value=0, **kwargs):
        super().__init__(**kwargs)
        self.prior_weight = float(os.getenv("PW", "5.0"))
        self.result_weight = float(os.getenv("RW", "1.0"))
        self.num_wins = (1. + action_value) / 2.
        self.num_losses = (1. - action_value) / 2.

    def expand(self, child_priors):
        """
        fake an action value
        :param child_priors:
        :return:
        """
        if self.is_expanded:
            return
        self.is_expanded = True
        offset = sum([p*p for p in child_priors.values()])
        for move, prior in child_priors.items():
            action_value = -self.Q() + self.prior_weight * (prior - offset)
            self.add_child(move, prior, action_value)

    def add_child(self, move, prior, action_value):
        board = self.board.copy()
        board.push_uci(move)
        self.children[move] = self.__class__(parent=self, move=move, prior=prior,
                                             board=board, action_value=action_value)

    def best_child(self):
        def beta(node):
            phi = self.result_weight * math.sqrt(math.sqrt(1 + node.parent.number_visits) / (1 + node.number_visits))
            return numpy.random.beta(node.num_wins/phi, node.num_losses/phi)
        return max(self.children.values(), key=beta)

    def backup(self, value_estimate: float):
        current = self
        turnfactor = 1
        current.num_wins += (1. - value_estimate * turnfactor) / 2.
        current.num_losses += (1. + value_estimate * turnfactor) / 2.
        current.number_visits += 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.num_wins += (1. - value_estimate * turnfactor) / 2.
            current.num_losses += (1. + value_estimate * turnfactor) / 2.

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda n: (n[1].num_wins - n[1].num_losses) / (n[1].num_wins+n[1].num_losses))
        if self.verbose:
            print(self.name, 'pv:', [(n[0],
                                      (n[1].num_wins - n[1].num_losses) / (n[1].num_wins+n[1].num_losses),
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

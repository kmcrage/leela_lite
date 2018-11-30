from search.uct import UCTNode
import numpy
import heapq
import os

"""
Taming Non-stationary Bandits: A Bayesian Approach
https://arxiv.org/pdf/1707.09727.pdf
"""


class Thompson_mixin:
    def __init__(self, action_value=0.,
                 prior_scale=float(os.getenv('PRIOR_SCALE', '.2')),
                 action_weight=float(os.getenv('ACTION_WEIGHT', '10')),
                 value_weight=float(os.getenv('VALUE_WEIGHT', '200')),
                 reward_weight=float(os.getenv('REWARD_WEIGHT', '2')),
                 discount_rate=float(os.getenv('DISCOUNT_RATE', '.995')),
                 **kwargs):
        super().__init__(**kwargs)
        self.prior_scale = prior_scale
        self.action_weight = action_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
        self.discount_rate = discount_rate
        # parent wins and losses, don't trust the action value
        self.num_wins = self.action_weight * (1. + action_value) / 2.
        self.num_losses = self.action_weight * (1. - action_value) / 2.

    def Q(self):
        """
        value from pov of the parent ie -1 is bad for parent
        :return:
        """
        return (self.num_wins - self.num_losses) / (1 + self.num_wins + self.num_losses)

    def expand(self, child_priors):
        """
        fake an action value, which we won't weight highly
        :param child_priors:
        :return:
        """
        if self.is_expanded:
            return
        self.is_expanded = True
        offset = sum([p*p for p in child_priors.values()])
        for move, prior in child_priors.items():
            action_value = numpy.clip(-self.Q() + self.prior_scale * (prior - offset), -1., 1.)
            self.add_child(move, prior, action_value)

    def add_child(self, move, prior, action_value):
        board = self.board.copy()
        board.push_uci(move)
        self.children[move] = self.__class__(parent=self, move=move, prior=prior,
                                             board=board, action_value=action_value)

    def best_child(self):
        """
        optimistic thompson: don't sample below the mean
        :return:
        """
        return max(self.children.values(),
                   key=lambda node: max((1 + node.Q()) / 2.,
                                        numpy.random.beta(1 + node.num_wins,
                                                          1 + node.num_losses)))

    def backup(self, value_estimate: float):
        current = self
        turnfactor = -1
        # trust the value estimate so we can reduce the variation a lot
        self.num_wins = self.value_weight * (1 - value_estimate)
        self.num_losses = self.value_weight * (1 + value_estimate)
        while current:
            # wins is parent wins
            for child in current.children.values():
                child.num_wins *= self.discount_rate
                child.num_losses *= self.discount_rate
            # trust the rewards, but variance is low by now
            current.num_wins += self.reward_weight * (1. + value_estimate * turnfactor) / 2.
            current.num_losses += self.reward_weight * (1. - value_estimate * turnfactor) / 2.
            current.number_visits += 1
            turnfactor *= -1
            current = current.parent

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda n: (n[1].Q(),
                                           n[1].number_visits))
        if self.verbose:
            print(self.name, 'pv:', [(n[0],
                                      n[1].Q(),
                                      .5 + n[1].Q()/2.,
                                      n[1].num_wins + n[1].num_losses,
                                      n[1].number_visits) for n in pv])
            # there could be no moves if we jump into a mate somehow
            print('prediction:', end=' ')
            predict = pv[0]
            while len(predict[1].children):
                predict = heapq.nlargest(1, predict[1].children.items(),
                                         key=lambda item: (item[1].Q(), item[1].number_visits))[0]
                print(predict[0], end=' ')
            print('')
        return pv[0] if pv else None


class UCTTNode(Thompson_mixin, UCTNode):
    name = 'uctt'


class UCTTMinusNode(UCTTNode):
    name = 'uctt_minus'

    def __init__(self, **kwargs):
        super().__init__(action_weight=10., prior_scale=.2, reward_weight=2., discount_rate=.993, **kwargs)


class UCTTPlusNode(UCTTNode):
    name = 'uctt_plus'

    def __init__(self, **kwargs):
        super().__init__(action_weight=10., prior_scale=.2, reward_weight=2., discount_rate=.997, **kwargs)

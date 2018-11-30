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
                 prior_weight=30., prior_scale=.2, reward_scale=20., discount_rate=.999,
                 **kwargs):
        super().__init__(**kwargs)
        self.prior_weight = float(os.getenv('PRIOR_WEIGHT', '20'))
        self.discount_rate = float(os.getenv('DISCOUNT_RATE', '.999'))
        self.prior_scale = float(os.getenv('PRIOR_SCALE', '.2'))
        self.reward_scale = float(os.getenv('REWARD_WEIGHT', '20'))
        # parent wins and losses
        self.prior_wins = self.prior_weight * (1. + action_value) / 2.
        self.prior_losses = self.prior_weight * (1. - action_value) / 2.
        self.num_wins = 0
        self.num_losses = 0

    def Q(self):
        """
        value from pov of the parent ie -1 is bad for parent
        :return:
        """
        if self.num_wins + self.num_losses > 0:
            return (self.num_wins - self.num_losses) / (self.num_wins + self.num_losses)
        return (self.prior_wins - self.prior_losses) / (self.prior_wins + self.prior_losses)

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
            action_value = numpy.clip(-self.Q() + self.prior_scale * (prior - offset), -1., 1.)
            self.add_child(move, prior, action_value)

    def add_child(self, move, prior, action_value):
        board = self.board.copy()
        board.push_uci(move)
        self.children[move] = self.__class__(parent=self, move=move, prior=prior,
                                             board=board, action_value=action_value)

    def best_child(self):
        print(' ')
        for node in self.children.values():
            eval = numpy.random.beta(1 + node.prior_wins + node.num_wins, 1 + node.prior_losses + node.num_losses)
            print(node.move, node.Q(), eval)
        return max(self.children.values(), key=lambda node: numpy.random.beta(1 + node.prior_wins + node.num_wins,
                                                                              1 + node.prior_losses + node.num_losses))

    def backup(self, value_estimate: float):
        current = self
        turnfactor = -1
        while current:
            # wins is parent wins
            for child in current.children.values():
                child.num_wins *= self.discount_rate
                child.num_losses *= self.discount_rate
                child.prior_wins *= self.discount_rate
                child.prior_losses *= self.discount_rate
            current.num_wins += self.reward_scale * (1. + value_estimate * turnfactor) / 2.
            current.num_losses += self.reward_scale * (1. - value_estimate * turnfactor) / 2.
            current.number_visits += 1
            turnfactor *= -1
            current = current.parent

    def outcome(self):
        print('root:', self.Q(), self.number_visits)
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda n: (n[1].Q(),
                                           n[1].number_visits))
        if self.verbose:
            print(self.name, 'pv:', [(n[0],
                                      n[1].Q(),
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
        super().__init__(prior_weight=3., prior_scale=.2, reward_scale=2., discount_rate=.999, **kwargs)


class UCTTPlusNode(UCTTNode):
    name = 'uctt_plus'

    def __init__(self, **kwargs):
        super().__init__(prior_weight=30., prior_scale=.2, reward_scale=2., discount_rate=.999, **kwargs)

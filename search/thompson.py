from search.uct import UCTNode
import numpy
import heapq


class Thompson_mixin:
    def __init__(self, action_value=0., prior_weight=30., prior_scale=.1, beta_scale=20., **kwargs):
        super().__init__(**kwargs)
        self.prior_scale = prior_scale
        self.beta_scale = beta_scale
        # parent wins and losses
        self.num_wins = 1 + prior_weight * (1. + action_value) / 2.
        self.num_losses = 1 + prior_weight * (1. - action_value) / 2.

    def Q(self):
        """
        value from pov of the parent ie -1 is bad for parent
        :return:
        """
        return (self.num_wins - self.num_losses) / (self.num_wins + self.num_losses)

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
        return max(self.children.values(), key=lambda node: numpy.random.beta(node.num_wins, node.num_losses))

    def backup(self, value_estimate: float):
        current = self
        turnfactor = -1
        # wins is parent wins
        current.num_wins += self.beta_scale * (1. + value_estimate * turnfactor) / 2.
        current.num_losses += self.beta_scale * (1. - value_estimate * turnfactor) / 2.
        current.number_visits += 1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.num_wins += self.beta_scale * (1. + value_estimate * turnfactor) / 2.
            current.num_losses += self.beta_scale * (1. - value_estimate * turnfactor) / 2.

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda n: (n[1].number_visits, n[1].Q()))
        if self.verbose:
            print(self.name, 'pv:', [(n[0], n[1].Q(), n[1].number_visits) for n in pv])
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


class UCTTMinusNode(UCTTNode):
    name = 'uctt_minus'

    def __init__(self, test_low=10, **kwargs):
        super().__init__(beta_scale=test_low, **kwargs)


class UCTTPlusNode(UCTTNode):
    name = 'uctt_plus'

    def __init__(self, test_high=30, **kwargs):
        super().__init__(beta_scale=test_high, **kwargs)

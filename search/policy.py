from search.uct import UCTNode
import math

"""
Adaptive policy 
See also exp3, https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf,
which is known to do well in adversarial contexts.

"""


class Policy_mixin:
    def __init__(self, temperature=3, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.policy = self.prior

    def U(self):  # returns float
        return math.sqrt(self.parent.number_visits) * self.policy / (1 + self.number_visits)

    def backup(self, value_estimate: float):
        """
        adapt policy as well as value
        compare with boltzman distribution
        :param value_estimate:
        :return:
        """
        current = self
        current.reward = -value_estimate
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current:
            # update policy before we update the mean (could be simpler just to set policy)
            current.policy *= math.exp((value_estimate * turnfactor - current.Q()) /
                                       ((1 + current.number_visits) * current.temperature))

            # ... and renormalise the children (we don't care about normalising root)
            if current.children:
                renorm = sum([c.policy for c in current.children.values()])
                for c in current.children.values():
                    c.policy /= renorm

            # update the mean of Q
            current.number_visits += 1
            current.total_value += value_estimate * turnfactor

            current = current.parent
            turnfactor *= -1


class PolicyUCTNode(Policy_mixin, UCTNode):
    name = 'policy'


class PolicyMinusNode(PolicyUCTNode):
    name = 'policyminus'

    def __init__(self, **kwargs):
        super().__init__(temperature=2.5, **kwargs)


class PolicyPlusNode(PolicyUCTNode):
    name = 'policyplus'

    def __init__(self, **kwargs):
        super().__init__(temperature=3.5, **kwargs)

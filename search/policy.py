from search.uct import UCTNode
import math

"""
Adaptive policy (see also exp3)
"""


class Policy_mixin:
    def __init__(self, temperature=2.2, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.policy = self.prior

    def U(self):  # returns float
        return math.sqrt(self.parent.number_visits) * self.policy / (1 + self.number_visits)

    def backup(self, value_estimate: float):
        """
        adapt policy as well as value
        :param value_estimate:
        :return:
        """
        current = self
        current.reward = -value_estimate
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current:
            current.number_visits += 1
            current.total_value += value_estimate * turnfactor
            current.policy *= math.exp((value_estimate * turnfactor - current.Q()) / current.temperature)

            if current.children:
                renorm = sum([c.policy for c in current.children.values()])
                for c in current.children.values():
                    c.policy /= renorm

            current = current.parent
            turnfactor *= -1


class PolicyUCTNode(Policy_mixin, UCTNode):
    name = 'policy'

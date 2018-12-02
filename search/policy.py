from search.uct import UCTNode
import math

"""
Adaptive policy 
See also exp3, https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf,
which is known to do well in adversarial contexts.

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
            # update policy before we update the mean
            K = len(current.parent.children) if current.parent and current.parent.children else 1
            current.policy *= math.exp(value_estimate * turnfactor / (K * current.policy * current.temperature))

            # ... and renormalise the children
            if current.children:
                renorm = sum([c.policy for c in current.children.values()])
                for c in current.children.values():
                    c.policy /= renorm

            current.number_visits += 1
            current.total_value += value_estimate * turnfactor

            # renorm root node to avoid policy going to zero
            if not current.parent:
                current.policy = 1
            current = current.parent
            turnfactor *= -1


class PolicyUCTNode(Policy_mixin, UCTNode):
    name = 'policy'


class PolicyMinusNode(PolicyUCTNode):
    name = 'policyminus'

    def __init__(self, **kwargs):
        super().__init__(temperature=.1, **kwargs)


class PolicyPlusNode(PolicyUCTNode):
    name = 'policyplus'

    def __init__(self, **kwargs):
        super().__init__(temperature=10, **kwargs)

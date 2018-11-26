import math
from search.uct import UCTNode


class Variance_mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_vsquared = self.parent.total_vsquared / self.parent.num_visits if parent else 0.   # float

    def sigma(self):  # returns float
        return math.sqrt(self.total_vsquared / (1 + self.number_visits) - self.Q() ** 2)

    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits) / (1 + self.number_visits) + self.prior) * self.sigma()

    def best_child(self):
        return max(self.children.values(),
                   key=lambda node: (node.Q() + self.cpuct * node.U))

    def backup(self, value_estimate: float):
        current = self
        current.reward = -value_estimate
        current.total_value += current.reward
        current.total_vsquared += current.reward ** 2
        current.number_visits += 1
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            current.total_value += (value_estimate * turnfactor)
            current.total_vsquared += value_estimate ** 2


class UCTV(Variance_mixin, UCTNode):
    name = 'uctv'

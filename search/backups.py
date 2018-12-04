from search.uct import UCTNode


class Product_mixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def backup(self, value_estimate: float):
        current = self
        current.reward = -value_estimate
        current.total_value += -value_estimate
        current.number_visits += 1

        current = current.parent
        while current:
            current.number_visits += 1
            q_vals = [1 - n.Q() for n in current.children.values() if n.number_visits]
            if any([v <= 0 for v in vals]):
                g_gmean = -1
            else:
                q_log_mean = sum([math.log(v) for v in q_vals]) / len(q_vals)
                g_gmean = exp(q_log_mean) - 1
            current.total_value = current.number_visits * q_gmean

            current = current.parent


class MinMax_mixin:
    def __init__(self, **kwargs):
        super(MinMax_mixin, self).__init__(**kwargs)
        self.minmax_threshhold = 50

    def backup(self, value_estimate: float):
        current = self
        current.reward = -value_estimate
        current.total_value += current.reward
        current.number_visits += 1
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current = current.parent
            current.number_visits += 1
            turnfactor *= -1
            if current.number_visits > self.minmax_threshhold:
                visited_children = [c for c in current.children.values() if c.number_visits]
                current.total_value = - (1 + current.number_visits) * max([c.Q() for c in visited_children])
            else:
                current.total_value += (value_estimate * turnfactor)


class MaxUct_mixin:
    def __init__(self, **kwargs):
        super(MaxUct_mixin, self).__init__(**kwargs)

    def backup_weight(self):
        return self.number_visits


class DPUCT_mixin:
    def __init__(self, **kwargs):
        super(DPUCT_mixin, self).__init__(**kwargs)

    def backup_weight(self):
        return self.prior

    def V(self):
        children = [n for n in self.children.values() if n.number_visits]
        if children:
            value = max([n.Q() for n in children])
        else:
            value = -self.Q()
        return value

    def backup(self, value_estimate: float):
        current = self
        current.reward = -value_estimate
        current.number_visits += 1
        while current.parent is not None:
            current = current.parent
            current.reward = 0
            # print('preupdate Q:', current.Q, len(current.children), current.number_visits)
            current.number_visits += 1
            # do we want to add in this reward? its more stable and will disappear with many evals
            current.total_value = 0
            sum_weights = 0
            for child in [n for n in current.children.values() if n.number_visits]:
                current.total_value += child.backup_weight() * child.V()
                sum_weights += child.backup_weight()
            current.total_value *= current.number_visits / sum_weights



class ProductUCTNode(Product_mixin, UCTNode):
    name = 'product'


class DPUCTNode(DPUCT_mixin, UCTNode):
    name = 'bellman'


class MaxUCTNode(MaxUct_mixin, DPUCT_mixin, UCTNode):
    name = 'maxuct'


class MinMaxUCTNode(MinMax_mixin, UCTNode):
    name = 'minmaxt'

from search.uct import UCTNode


class PrunedUCT_mixin:
    def __init__(self, **kwargs):
        super(PrunedUCT_mixin, self).__init__(**kwargs)
        self.prune = 0.05

    def expand(self, child_priors):
        self.is_expanded = True
        if not child_priors:
            return
        threshold = max(child_priors.values()) * self.prune
        for move, prior in child_priors.items():
            if prior > threshold:
                self.add_child(move, prior)

class PrunedUCTNode(PrunedUCT_mixin, UCTNode):
    name = 'pruned'

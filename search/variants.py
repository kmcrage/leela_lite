from search.uct import UCTNode


class Cutoff_mixin:
    def __init__(self, cutoff=0.5, **kwargs):
        super(Cutoff_mixin, self).__init__(**kwargs)
        self.cutoff = cutoff

    def select_leaf(self):
        current = self
        # don't bother confirming the win if we boom,
        # but watch out for draws
        min_reward = max(0., current.reward - self.cutoff)
        max_reward = min(0., current.reward + self.cutoff)
        while current.is_expanded and current.children and min_reward < current.reward < max_reward:
            current = current.best_child()
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current


class CutoffUCTNode(Cutoff_mixin, UCTNode):
    name = 'cutoff'

from search.uct import UCTNode

"""
http://www.ru.is/~yngvi/pdf/GudmundssonB11.pdf
Sufficiency Threshold
"""


class ST_mixin:
    def __init__(self, **kwargs):
        super(ST_mixin, self).__init__(**kwargs)
        self.win_threshhold = 0.25

    def best_child(self):
        c = 0 if any([c.Q() > self.win_threshhold for c in self.children.values()]) else 1
        return max(self.children.values(),
                   key=lambda node: node.Q() + c * self.cpuct * node.U())


class STUCTNode(ST_mixin, UCTNode):
    name = 'st'

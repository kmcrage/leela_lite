from search.uct import UCTNode
import heapq
import numpy

"""
http://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf
Score Bounded Monte-Carlo Tree Search

"""


class Bounded_mixin:
    def __init__(self, bound_penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.bound_penalty = bound_penalty
        self.alpha = -1
        self.beta = 1

    def Q(self):  # returns float
        return numpy.clip(self.total_value / (1 + self.number_visits), self.alpha, self.beta)

    def best_child(self):
        """
        prune the children with the bounds we've generated
        :return:
        """
        candidates = [c for c in self.children.values() if c.alpha <= c.beta and c.beta > -self.beta]
        # if beta=alpha, its solved, so just choose one solution and stick with it
        if not candidates:
            candidates = [c for c in self.children.values() if c.beta == -self.beta][:1]

        return max(candidates,
                   key=lambda node: node.Q() + node.cpuct * node.U() + node.bound_penalty * node.alpha)

    def backup(self, value_estimate: float):
        """
        additional backup of optimistic and pessimistic bounds
        clip Q to fit the bounds
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

            if not current.children:
                # this is a terminal node
                current.beta = value_estimate * turnfactor
                current.alpha = value_estimate * turnfactor
            else:
                current.beta = -max([c.alpha for c in current.children.values()])
                current.alpha = -max([c.beta for c in current.children.values()])

            current = current.parent
            turnfactor *= -1

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children.items(),
                            key=lambda item: (item[1].number_visits, item[1].Q()))
        if self.verbose:
            print(self.name, 'pv:', [(n[0],
                                      n[1].Q(),
                                      '[%f, %f]' % (n[1].alpha, n[1].beta),
                                      n[1].number_visits) for n in pv])
        # there could be no moves if we jump into a mate somehow
        print('prediction:', end=' ')
        predict = pv[0]
        while len(predict[1].children):
            predict = heapq.nlargest(1, predict[1].children.items(),
                                     key=lambda item: (item[1].number_visits, item[1].Q()))[0]
            print(predict[0], end=' ')
        print('')
        return pv[0] if pv else None


class BoundedUCTNode(Bounded_mixin, UCTNode):
    name = 'bounded'

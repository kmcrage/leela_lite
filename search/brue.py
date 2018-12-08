import weakref
import heapq

class BRUENode:
    name = 'brue'

    def __init__(self, board=None, parent=None, prior=0, move=None, verbose=True):
        self.board = board
        self._parent = weakref.ref(parent) if parent else None
        self.children = []
        self.move = move
        self.is_expanded = False
        self.prior = prior         # float
        self.q = -parent.q if parent else 0
        self.q_sse = 0
        self.number_visits = 0     # int
        self.verbose = verbose

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def var(self):
        if self.number_visits < 3:
            return 1
        return self.q_sse / self.number_visits

    def ev(self):
        exploit = self.exploitation()
        return sum([c.prior * c.var() for c in self.children if c != exploit]) / exploit.prior

    def ve(self):
        current = self
        var_sum = current.var()
        depth = 1
        while current.children:
            current = current.exploitation()
            depth += 1
            var_sum += current.var()
        return var_sum / depth

    def exploitation(self):
        return max(self.children, key=lambda node: node.q)
    
    def exploration(self):
        exploit = self.exploitation()
        return max([c for c in self.children if c != exploit],
                   key=lambda node: node.prior * node.var())

    def expand(self, child_priors):
        if self.is_expanded:
            return
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        board = self.board.copy()
        board.push_uci(move)
        self.children.append(self.__class__(parent=self, prior=prior, move=move, board=board))

    def get_node(self, move):
        if move in self.children:
            return self.children[move]
        return None

    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        sample = -value_estimate
        while current:
            current.number_visits += 1
            e = sample - current.q
            current.q += e / current.number_visits
            current.q_sse += e * (sample - current.q)

            current = current.parent
            sample *= -1

    def best_child(self):
        # print(self.move, 'explore', self.ev(), 'exploit', self.ve())
        if self.ev() > self.ve():
            return self.exploration()
        else:
            return self.exploitation()

    def select_leaf(self):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child()
        return current

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children,
                            key=lambda item: (item.q, item.number_visits))
        if self.verbose:
            print(self.name, 'pv:', [(n.move, n.q, n.var(), n.number_visits) for n in pv])
        current = self.exploitation()
        print('brue prediction:', end=' ')
        while current.children:
            current = current.exploitation()
            print(current.move, current.var(), end=', ')
        print('')
        result = self.exploitation()
        return result.move, result

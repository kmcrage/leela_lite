import weakref


class BRUENode:
    def __init__(self, board=None, parent=None, prior=0, move=None):
        self.board = board
        self._parent = weakref.ref(parent) if parent else None
        self.children = []
        self.move = move
        self.is_expanded = False
        self.prior = prior         # float
        self.q = -1
        self.q_sse = 0
        self.number_visits = 0     # int

    @property
    def parent(self):
        return self._parent() if self._parent else None

    def var(self):
        if self.number_visits < 2:
            return 1
        return self.q_sse / self.number_visits - self.q * self.q

    def ev(self):
        return sum([c.prior * c.var() for c in self.children])

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
        return max(self.children, key=lambda node: node.prior * node.var)
    
    def expand(self, child_priors):
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

    def update_node(self, reward):
        self.number_visits += 1
        self.q += (reward - self.q) / self.number_visits

    def best_child(self):
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
        current = self
        while current.children:
            current = current.exploitation()
            print((current.move, current.number_visits), end=', ')
        print('')
        result = self.exploitation()
        return result.move, result

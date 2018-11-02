from random import choices
from collections import OrderedDict
import math


class BRUENode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior         # float
        self.q = 0.
        self.number_visits = 0     # int
        self.uncertainty = .15
        
    def exploitation(self):
        return max(self.children.values(), key=lambda node: node.q)
    
    def exploration(self):
        children = self.children
        return choices(list(children.values()),
                       [node.prior for node in children.values()], k=1)[0]
    
    def expand(self, child_priors):
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        child = self.build_child(move)
        self.children[move] = BRUENode(child, parent=self, prior=prior)

    def build_child(self, move):
        board = self.board.copy()
        board.push_uci(move)
        return board

    @staticmethod
    def switch_function(num, _):
        return 1 + num % int(1 + math.log(1 + num))

    @staticmethod
    def end_of_probe(node, net, _):
        if node.children:
            return False
        if not node.board.pc_board.is_game_over():
            child_priors, value_estimate = net.evaluate(node.board)
            node.expand(child_priors)
            node.q = value_estimate
            node.number_visits = 1
        return True

    def update_node(self, reward):
        self.number_visits += 1
        self.q += (reward - self.q) / self.number_visits


class Mcts2e:
    def __init__(self, net):
        self.net = net

    def probe(self, node, depth, switch):
        if node.end_of_probe(node, self.net, depth):
            if switch > depth:
                switch = depth
            reward = node.q
        else:
            if depth < switch:
                child = node.exploration()
            else:
                child = node.exploitation()
            reward = node.q - self.probe(child, depth+1, switch)
        print('node q:', node.q, 'depth', depth)
        if depth == switch:
            print('update depth', depth, 'reward', reward)
            node.update_node(reward)
        return reward

    def result(self, root=None, num_reads=0):
        switch = 0
        for n in range(num_reads):
            switch = root.switch_function(n, switch)
            self.probe(root, 0, switch)
            print(sorted([(i[0], i[1].q) for i in root.children.items()], key=lambda item: item[1]))
        return max(root.children.items(),
                   key=lambda item: (item[1].q, item[1].number_visits))


def BRUE_search(board, num_reads, net=None, **_):
    root = BRUENode(board)
    search = Mcts2e(net)
    return search.result(root, num_reads)

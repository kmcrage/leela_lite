import math
import heapq
from collections import OrderedDict
import chess.polyglot

"""
UCD: UCT with transpositions, and values on edges

https://hal.archives-ouvertes.fr/hal-01499672/document


UCD : Upper confidence bound for rooted directed acyclic graphs
Tristan Cazenave, Jean Méhat, Abdallah Saffidine

TODO: cache management
"""


NODE_CACHE = {}  # Dict [hash_of_board, UCDNode]


class UCDRollout:
    def __init__(self, root):
        self.history = []  # [edge, edge, ...]
        self.root = root

    def leaf_node(self):
        if self.history:
            self.history[-1].update_child()  # only do this if we follow the node
            node = self.history[-1].child
        else:
            node = self.root
        return node

    def backup(self, value):
        if len(self.history) and self.history[-1] in self.history[:-1]:
            reward = 0  # closed loop
        else:
            reward = -value
        if self.history:
            self.history[-1].terminal_value += reward
            self.history[-1].terminal_visits += 1
        while self.history:
            edge = self.history.pop()
            edge.total_value += reward
            edge.number_visits += 1
            reward *= -1


class UCDEdge:
    def __init__(self, parent=None, move=None, prior=0,
                 cpuct=3.4):
        self.cpuct = cpuct
        self.d1 = 1
        self.d2 = 0
        self.d3 = 0

        self.parent = parent
        self.move = move
        self.child = None

        self.prior = prior  # float
        self.total_value = 0.  # float
        self.number_visits = 0  # int
        self.terminal_value = 0.  # float
        self.terminal_visits = 0  # int

    def update_child(self):
        board = self.parent.board.copy()
        board.push_uci(self.move)
        if self.child:
            # update board so we track repetition
            self.child.board = board
            return

        zhash = chess.polyglot.zobrist_hash(board.pc_board)
        if zhash not in NODE_CACHE:
            NODE_CACHE[zhash] = self.parent.__class__(board=board)
        else:
            NODE_CACHE[zhash].board = board
        self.child = NODE_CACHE[zhash]
        # is this already a parent of this node?
        # ie are we entering a cycle?
        # this may be paranoia
        if not any([self.parent == e.parent for e in self.child.parents]):
            self.child.parents.append(self)

    def n(self, depth):
        if not depth:
            return self.number_visits

        visits = self.terminal_visits
        for edge in [e for e in self.child.children if e.number_visits]:
            visits += edge.n(depth - 1)
        return visits

    def p(self, depth):
        visits = 0
        for edge in [e for e in self.parent.children if e.number_visits]:
            visits += edge.n(depth)
        return visits

    def mu(self, depth):
        if not depth:
            return self.total_value / max(1, self.number_visits)
        visits = self.terminal_visits
        value = self.terminal_value
        if self.child:
            for edge in [e for e in self.child.children if e.number_visits]:
                value += edge.mu(depth-1) * edge.number_visits
                visits += edge.number_visits
        return value / max(1, visits)

    def Q(self):
        return self.mu(self.d1)

    def U(self):
        return self.prior * math.sqrt(self.p(self.d2)) / max(1, self.n(self.d3))


class UCDNode:
    name = 'ucd'
    edge_class = UCDEdge
    rollout_class = UCDRollout

    def __init__(self, board=None):
        self.board = board
        self.parents = []  # [UCDNode]
        self.children = []  # [UCDNode]

    def best_edge(self):
        return max(self.children,
                   key=lambda edge: edge.Q() + edge.cpuct * edge.U())

    def get_node(self, move):
        for edge in self.children:
            if move == edge.move:
                return edge.child
        return None

    def generate_rollout(self):
        rollout = self.rollout_class(root=self)
        current = self
        while current and current.children and current not in rollout.history[:-1]:
            edge = current.best_edge()
            rollout.history.append(edge)
            current = edge.child
            # make sure the board is passed on to avoid loops
            if current:
                edge.update_child()

        return rollout

    def expand(self, child_priors):
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children.append(self.edge_class(parent=self, move=move, prior=prior))

    def outcome(self):
        size = min(5, len(self.children))
        pv = heapq.nlargest(size, self.children,
                            key=lambda item: (item.number_visits, item.Q()))

        print(self.name, 'pv:', [(n.move, n.Q(), n.U(), n.number_visits) for n in pv])
        return [pv[0].move, pv[0].child]

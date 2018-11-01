import numpy as np
import math
from random import choices
import lcztools
from lcztools import LeelaBoard
import chess
from collections import OrderedDict

MAX_DEPTH = 10

class BRUENode():
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior         # float
        self.total_value = 0. # float
        self.number_visits = 0     # int
        self.uncertainty = .15

    def Q(self):
        return self.total_value/self.number_visits
        
    def exploit(self):
        children = self.children
        return max(self.children.values(), key=lambda node: node.prior+0.)
    
    def explore(self):
        children = self.children
        return choices(list(children.values()), [node.prior for node in children.values()], k=1)[0]
    
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
        
    def backup(self, leaf, value_estimate: float):
        current = leaf
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current != self:
            current = current.parent
            value_estimate *= turnfactor
        self.prior = max(0.0, (self.number_visits * self.prior + value_estimate) / (self.number_visits + 1))
        self.total_value += value_estimate

    
    def dump(self, move, C):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q())
        print("U: ", self.U())
        print("BestMove: ", self.Q() + C * self.U())
        #print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")

def BRUE_search(board, num_reads, net=None, C=1.0):
    assert(net != None)
    root = BRUENode(board)
    n = 0
    probe = 0
    while n < num_reads:
        switchingPoint = probe % MAX_DEPTH
        # print('probe:', probe, 'evals:', n, 'switch: ', switchingPoint)
        level = 0
        current = root
        update_node = None
        while level < MAX_DEPTH and n < num_reads:
            if not current.number_visits:
                child_priors, reward = net.evaluate(current.board)
                n += 1
                if not child_priors:
                    break
                current.expand(child_priors)


            if level < switchingPoint:
                current = current.explore()
                # print('explore', level+1, current.number_visits)
            else:
                if not update_node:
                    update_node = current
                current = current.exploit()
                # print('exploit', level+1, current.number_visits)
            level += 1


        _, reward = net.evaluate(current.board)
        update_node.number_visits += 1
        update_node.backup(current, reward)
        probe += 1
    
    #for action, child in root.children.items():
    #    print(action, child.number_visits, child.Q())
    return max(root.children.items(), key=lambda item: item[1].Q())



#num_reads = 10000
#import time
#tick = time.time()
#UCT_search(GameState(), num_reads)
#tock = time.time()
#print("Took %s sec to run %s times" % (tock - tick, num_reads))
#import resource
#print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

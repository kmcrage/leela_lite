from search.neural_net import NeuralNet
from search.uct import UCTNode
from search.ucd import UCDNode
from search.crazy import CRAZY_search
from search.brue import BRUE_search
from search.voi import VOI_search
from search.mpa_backup import MPA_search
from search.backups import DPUCTNode, ProductUCTNode
from search.backups import MaxUCTNode
from search.backups import MinMaxUCTNode
from search.variants import CutoffUCTNode
from search.prune import PrunedUCTNode
from search.minmax_backup import MinMax_search
from search.srcr import SRCR_search
from search.asymmetric import Asym_search
from search.uctv import UCTVNode
from search.thompson import UCTTNode
from search.sota import SOTA_search
from search.st import STUCTNode
from search.normal import NSNode, NSMinusNode, NSPlusNode
from search.bayes import BayesNode, BayesMinusNode, BayesPlusNode
from search.bounds import BoundedUCTNode
from search.policy import PolicyUCTNode, PolicyMinusNode, PolicyPlusNode

from search.alphabeta import ABNode
from search.alphabeta_uct import ABUCTNode

from functools import partial
from search.mcds import mcds_search
from search.mcts import mcts_search, mcts_eval_search
from search.nse import nse_search

# active searches first
#
#
engines = {'uct': partial(mcts_search, UCTNode),
           'dpuct': partial(mcts_search, DPUCTNode),
           'maxuct': partial(mcts_search, MaxUCTNode),
           'product': partial(mcts_search, ProductUCTNode),
           'minmaxt': partial(mcts_search, MinMaxUCTNode),
           'pruned': partial(mcts_search, PrunedUCTNode),
           'st': partial(mcts_search, STUCTNode),
           'uctv': partial(mcts_search, UCTVNode),
           'uctt': partial(mcts_search, UCTTNode),

           'ns': partial(mcts_search, NSNode),
           'nsminus': partial(mcts_search, NSMinusNode),
           'nsplus': partial(mcts_search, NSPlusNode),

           'bayes': partial(mcts_search, BayesNode),
           'bayesminus': partial(mcts_search, BayesMinusNode),
           'bayesplus': partial(mcts_search, BayesPlusNode),

           'bounded': partial(mcts_search, BoundedUCTNode),
           'policy': partial(mcts_search, PolicyUCTNode),
           'policyminus': partial(mcts_search, PolicyMinusNode),
           'policyplus': partial(mcts_search, PolicyPlusNode),

           'ucd': partial(mcds_search, UCDNode),
           'alphabeta': partial(mcts_search, ABNode),
           'ab_uct': partial(mcts_search, ABUCTNode),

           'nse_uct': partial(nse_search, UCTNode),

           'cutoff': partial(mcts_eval_search, CutoffUCTNode),

           'mpa': MPA_search,
           'voi': VOI_search,
           'crazy': CRAZY_search,
           'srcr': SRCR_search,
           'asym': Asym_search,
           'sota': SOTA_search,

           'human': 'brain'
           }

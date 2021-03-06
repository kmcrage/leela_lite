from search.neural_net import NeuralNet
from search.uct import UCTNode, AdaptNode
from search.crazy import CRAZY_search
from search.brue import BRUE_search
from search.voi import VOINode
from search.mpa_backup import MPA_search
from search.backups import DPUCTNode, MaxUCTNode
from search.minmax_backup import MinMax_search
from search.srcr import SRCR_search
from search.asymmetric import AsymNode
from search.uctv import UCTV_search
from search.sota import SOTA_search

from functools import partial
from search.mcts import mcts_search

# active searches first
#
#
engines = {'uct': partial(mcts_search, UCTNode),
           'dpuct': partial(mcts_search, DPUCTNode),
           'maxuct': partial(mcts_search, MaxUCTNode),
           'adapt': partial(mcts_search, AdaptNode),

           'asym': partial(mcts_search, AsymNode),
           'voi': partial(mcts_search, VOINode),

           'mpa': MPA_search,
           'uctv': UCTV_search,
           'crazy': CRAZY_search,
           'srcr': SRCR_search,
           'sota': SOTA_search,

           'human': 'brain'
           }

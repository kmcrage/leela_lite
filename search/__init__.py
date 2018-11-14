from search.neural_net import NeuralNet
from search.uct import UCTNode
from search.ucd import UCDNode
from search.crazy import CRAZY_search
from search.brue import BRUE_search
from search.voi import VOI_search
from search.mpa_backup import MPA_search
from search.backups import DPUCTNode
from search.backups import MaxUCTNode
from search.minmax_backup import MinMax_search
from search.srcr import SRCR_search
from search.asymmetric import Asym_search
from search.uctv import UCTV_search
from search.sota import SOTA_search

from functools import partial
from search.mcds import mcds_search
from search.mcts import mcts_search

# active searches first
#
#
engines = {'uct': partial(mcts_search, UCTNode),
           'shared': partial(mcts_search, UCTSNode),
           'dpuct': partial(mcts_search, DPUCTNode),
           'maxuct': partial(mcts_search, MaxUCTNode),

           'ucd': partial(mcds_search, UCDNode),

           'mpa': MPA_search,
           'uctv': UCTV_search,
           'crazy': CRAZY_search,
           'srcr': SRCR_search,
           'asym': Asym_search,
           'sota': SOTA_search,

           'human': 'brain'
           }

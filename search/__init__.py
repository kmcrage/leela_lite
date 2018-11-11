from search.neural_net import NeuralNet
from search.uct import UCTNode
from search.crazy import CRAZY_search
from search.brue import BRUE_search
from search.voi import VOI_search
from search.mpa_backup import MPA_search
from search.backups import BellmanNode
from search.backups import MaxUctNode
from search.minmax_backup import MinMax_search
from search.srcr import SRCR_search
from search.asymmetric import Asym_search
from search.uctv import UCTV_search
from search.sota import SOTA_search

from functools import partial
from search.mcts import mcts_search

engines = {'uct': partial(mcts_search, UCTNode),
           'bellman': partial(mcts_search, BellmanNode),
           'maxuct': partial(mcts_search, MaxUctNode),
           'mpa': MPA_search,
           'uctv': UCTV_search,
           'srcr': SRCR_search,
           'asym': Asym_search,
           'sota': SOTA_search,
           'human': 'brain'
           }

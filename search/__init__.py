from search.neural_net import NeuralNet
from search.uct import UCT_search
from search.crazy import CRAZY_search
from search.brue import BRUE_search
from search.voi import VOI_search
from search.mpa_backup import MPA_search
from search.bellman_backup import Bellman_search
from search.minmax_backup import MinMax_search
from search.srcr import SRCR_search
# from uct.util import softmax, temp_softmax


engines = {'uct': UCT_search,
           'minmax': MinMax_search,
           'bellman': Bellman_search,
           'mpa': MPA_search,
           'srcr': SRCR_search,
           'human': 'brain'
           }

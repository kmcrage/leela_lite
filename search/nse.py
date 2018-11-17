import math
import search.mcts
import heapq

"""
https://arxiv.org/pdf/1609.02606.pdf

Nonlinear Sequential Elimination
"""

def c(p, num_moves):
    result = math.pow(2, -p)
    for r in range(2, num_moves + 1):
        result += math.pow(r, -p)
    return result


def n(p, budget, num_moves, rnd):
    if not rnd:
        return 0  # round zero is the baseline with no evals
    result = budget - num_moves
    result /= c(p, num_moves) * math.pow(num_moves - rnd + 1, p)
    return math.ceil(result)


def nse_search(nodeclass, board, budget, net=None, root=None, p=1.5, verbose=True):
    assert(net is not None)
    if not root:
        root = nodeclass(board=board)

    if not root.children:
        child_priors, value_estimate = net.evaluate(root.board)
        root.expand(child_priors)
        budget -= 1

    G = list(root.children.values())
    K = len(G)
    for r in range(1, K):
        child_budget = n(p, budget, K, r) - n(p, budget, K, r - 1)
        if verbose:
            print('round', r, 'budget', child_budget)
        if child_budget:
            for child in G:
                search.mcts.mcts_search(nodeclass, board, child_budget, net=net, root=child, verbose=False)
        worst = heapq.nsmallest(1, G, key=lambda ch: (ch.Q(), ch.prior))[0]
        if len(G) < 5 and verbose:
            print('worst', worst.move, worst.Q())
            print([(ch.move, ch.Q()) for ch in G])
        G.remove(worst)

    return G[0].move, G[0]  # only remaining member

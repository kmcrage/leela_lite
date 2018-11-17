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
    result = budget - num_moves
    result /= c(p, num_moves) * math.pow(num_moves - rnd + 1, p)
    return math.ceil(result)


def nse_search(nodeclass, board, budget, net=None, root=None, p=1.5):
    assert(net is not None)
    if not root:
        root = nodeclass(board=board)

    if not root.children:
        child_priors, value_estimate = net.evaluate(root.board)
        root.expand(child_priors)
        budget -= 1

    G = list(root.children.values())
    K = len(G)
    print('budgets')
    b = 0
    for r in range(1, K):
        b += (K - r + 1) * n(p, budget, K, r)
        print('round', r, 'budget', n(p, budget, K, r), 'total',  b)
    for r in range(1, K):
        child_budget = n(p, budget, K, r) - n(p, budget, K, r - 1)
        if child_budget == 0:
            child_budget = 1
            budget -= K
        print('round', r, 'budget', child_budget)
        for child in G:
            search.mcts.mcts_search(nodeclass, board, child_budget, net=net, root=child)
        worst = heapq.nsmallest(1, G, key=lambda c: (c.Q(), -c.number_visits))[0]
        print('worst', worst.move)
        G.remove(worst)
        print([(ch.move, ch.Q()) for ch in G])

    return G[0]  # only remaining member

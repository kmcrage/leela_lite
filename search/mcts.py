

def mcts_search(nodeclass,
                board, budget, net=None, root=None,
                verbose=True):
    assert(net is not None)
    if not root:
        root = nodeclass(board=board)
    root.verbose = verbose
    while budget > 0:
        leaf = root.select_leaf()
        child_priors, value_estimate = net.evaluate(leaf.board)  # cached
        if not leaf.is_expanded:
            leaf.expand(child_priors)
            budget -= 1
        leaf.backup(value_estimate)

    return root.outcome()


def mcts_eval_search(nodeclass,
                     board, budget, net=None, root=None,
                     verbose=True, rollout_cost=0.01):
    assert(net is not None)
    if not root:
        root = nodeclass(board=board)
    root.verbose = verbose
    while budget > 0:
        leaf = root.select_leaf()
        if leaf.is_expanded:
            leaf.backup(-leaf.reward)
            budget -= rollout_cost
        else:
            child_priors, value_estimate = net.evaluate(leaf.board)
            leaf.expand(child_priors)
            leaf.backup(value_estimate)
            budget -= 1

    return root.outcome()

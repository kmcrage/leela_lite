
def mcts_search(nodeclass, board, num_reads, net=None, root=None):
    assert(net is not None)
    if not root:
        root = nodeclass(board=board)
    for _ in range(num_reads):
        rollout = root.generate_rollout()
        child_priors, value_estimate = net.evaluate(rollout.history[-1].board)
        rollout.expand(child_priors)
        rollout.backup(value_estimate)

    return root.outcome()

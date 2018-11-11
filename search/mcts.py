
def mcts_search(nodeclass, board, num_reads, net=None, root=None):
    assert(net is not None)
    if not root:
        root = nodeclass(board)
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)

    return root.outcome()

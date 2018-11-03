#!/usr/bin/python3
import sys
sys.path.extend(['/content/lczero_tools/src', '/content/python-chess', '/content/leela-lite'])
from lcztools import load_network, LeelaBoard
import search
import chess
import chess.pgn
import time

if len(sys.argv) != 4:
    print("Usage: python3 leela_lite.py <backend> <weights file> <nodes>")
    print(len(sys.argv))
    exit(1)

backend = sys.argv[1]
weights = sys.argv[2]
nodes = int(sys.argv[3])


board = LeelaBoard()

net = load_network(backend=backend, filename=weights, policy_softmax_temp=2.2)
nn = search.NeuralNet(net=net)
# policy, value = net.evaluate(board)
# print(policy)
# print(value)
# print(uct.softmax(policy.values()))

SELFPLAY = True

while True:
    if not SELFPLAY:
        print(board)
        print("Enter move: ", end='')
        sys.stdout.flush()
        line = sys.stdin.readline()
        line = line.rstrip()
        board.push_uci(line)
    print(board)
    print("thinking...")
    start = time.time()
    best, node = search.UCT_search(board, nodes, net=nn, C=3.4)
    elapsed = time.time() - start
    print("UCT best: ", best, node.Q())
    print("Time: {:.3f} nps".format(nodes/elapsed))
    #print(nn.evaluate.cache_info())
    board.push_uci(best)
    if board.pc_board.is_game_over() or board.is_draw():
        print("Game over... result is {}".format(board.pc_board.result(claim_draw=True)))
        print(board)
        print(chess.pgn.Game.from_board(board.pc_board))
        break
    print(board)
    print("thinking...")
    start = time.time()
    best, node = search.VOI_search(board, nodes, net=nn)
    elapsed = time.time() - start
    print("VOI best: ", best, node.Q)
    print("Time: {:.3f} nps".format(nodes/elapsed))
    #print(nn.evaluate.cache_info())
    board.push_uci(best)
    if board.pc_board.is_game_over() or board.is_draw():
        print("Game over... result is {}".format(board.pc_board.result(claim_draw=True)))
        print(board)
        print(chess.pgn.Game.from_board(board.pc_board))
        break


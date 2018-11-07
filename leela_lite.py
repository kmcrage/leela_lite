#!/usr/bin/python3
import sys
sys.path.extend(['/content/lczero_tools/src', '/content/python-chess', '/content/leela-lite'])
from lcztools import load_network, LeelaBoard
import search
import chess
import chess.pgn
import time
from os import path

engines = {'uct': search.UCT_search,
           'minmax': search.MinMax_search,
           'bellman': search.Bellman_search,
           'mpa': search.MPA_search,
           'srcr': search.SRCR_search
           }

if len(sys.argv) != 6:
    print("Usage: python3 leela_lite.py <policy1> <policy2> <weights file> <nodes> <c>")
    print(len(sys.argv))
    exit(1)

players = [sys.argv[1], sys.argv[2]]
weights = sys.argv[3]
nodes = int(sys.argv[4])
c = float(sys.argv[5])


backend = 'pytorch_cuda' if path.exists('/opt/bin/nvidia-smi') else 'pytorch_cpu'
board = LeelaBoard()
net = load_network(backend=backend, filename=weights, policy_softmax_temp=2.2)
nn = search.NeuralNet(net=net)


turn = 0
while True:
    print(board)
    print("thinking...")
    start = time.time()
    best, node = engines[players[turn]](board, nodes, net=nn, C=c)
    elapsed = time.time() - start
    print(board.pc_board.fullmove_number, players[turn], "best: ", best)
    print("Time: {:.3f} nps".format(nodes/elapsed))
    board.push_uci(best)
    if board.pc_board.is_game_over() or board.is_draw():
        print("Game over... result is {}".format(board.pc_board.result(claim_draw=True)))
        print(board)
        print(chess.pgn.Game.from_board(board.pc_board))
        break
    turn = 1 - turn

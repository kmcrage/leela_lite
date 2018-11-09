#!/usr/bin/python3
import sys
sys.path.extend(['/content/lczero_tools/src', '/content/python-chess', '/content/leela-lite'])
from lcztools import load_network, LeelaBoard
import search
import chess.pgn
import time
from os import path


if len(sys.argv) != 6:
    print("Usage: python3 leela_lite.py <policy1> <policy2> <weights file> <nodes>")
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
    if players[turn] == "human":
        print("Enter move: ", end='')
        sys.stdout.flush()
        line = sys.stdin.readline()
        line = line.rstrip()
        board.push_uci(line)
    else:
        print("thinking...")
        start = time.time()
        search.engines['uct'](board, nodes, net=nn)
        best, node = search.engines[players[turn]](board, nodes, net=nn)
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

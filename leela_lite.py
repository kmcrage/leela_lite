#!/usr/bin/python3
import argparse
import chess.pgn
from lcztools import load_network, LeelaBoard
import os.path
import search
import sys
import time

default_engine = 'uct'

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--weights",
                    help="a path to a weights file")
parser.add_argument("-w", "--white",
                    help="the engine to use for white",
                    choices=search.engines.keys(), default=default_engine)
parser.add_argument("-b", "--black",
                    help="the engine to use for black",
                    choices=search.engines.keys(), default=default_engine)
parser.add_argument("-k", "--kibitz",
                    help="the engine to use for kibitz",
                    choices=search.engines.keys()+['none'], default=default_engine)
parser.add_argument("-n", "--nodes",
                    help="the engine to use for black",
                    type=int, default=800)
parser.add_argument("-v", "--verbosity", action="count", default=0)
args = parser.parse_args()

backend = 'pytorch_cuda' if os.path.exists('/opt/bin/nvidia-smi') else 'pytorch_cpu'
net = load_network(backend=backend, filename=args.weights, policy_softmax_temp=2.2)
nn = search.NeuralNet(net=net)
board = LeelaBoard()

players = [{'engine': args.white,
            'root': None,
            'resets': 0},
           {'engine': args.black,
            'root': None,
            'resets': 0}]

turn = 0
while True:
    print(board)
    if players[turn] == "human":
        print("Enter move: ", end='')
        sys.stdout.flush()
        line = sys.stdin.readline()
        best = line.rstrip()
    else:
        if args.verbosity:
            print("thinking...")
            if players[turn]['root'] and hasattr(players[turn]['root'], 'number_visits'):
                print('starting with', players[turn]['root'].number_visits, 'visits')
        start = time.time()
        if players[turn]['engine'] != args.kibitz and args.kibitz != 'none':
            search.engines[args.kibitz](board, args.nodes, net=nn)
        best, node = search.engines[players[turn]['engine']](board, args.nodes,
                                                             net=nn, root=players[turn]['root'])
        print(board.pc_board.fullmove_number, players[turn]['engine'], "best: ", best)
        elapsed = time.time() - start
        if args.verbosity:
            print("Time: {:.3f} nps".format(args.nodes/elapsed))
        players[turn]['root'] = node

    board.push_uci(best)
    if players[1 - turn]['root']:
        players[1 - turn]['root'] = players[1-turn]['root'].get_node(best)
    if not players[1 - turn]['root']:
        if args.verbosity:
            print('tree reset for player', 1-turn, players[1 - turn]['engine'])
        players[1 - turn]['resets'] += 1
        players[1 - turn]['root'] = None

    if board.pc_board.is_game_over() or board.is_draw():
        print("Game over... result is {}".format(board.pc_board.result(claim_draw=True)))
        print(board)
        print(chess.pgn.Game.from_board(board.pc_board))
        break
    turn = 1 - turn

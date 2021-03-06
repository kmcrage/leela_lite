import sys
sys.path.extend(['/content/lczero_tools/src', '/content/python-chess', '/content/leela-lite'])
from lcztools import load_network, LeelaBoard
import search
from os import path

logfile = open("leelalite_uct.log", "w")
LOG = False


def log(str):
    if LOG:
        logfile.write(str)
        logfile.write("\n")
        logfile.flush()


def send(str):
    log(">{}".format(str))
    sys.stdout.write(str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def process_position(tokens):
    board = LeelaBoard()

    offset = 0

    if tokens[1] == 'startpos':
        offset = 2
    elif tokens[1] == 'fen':
        fen = " ".join(tokens[2:8])
        board = LeelaBoard(fen=fen)
        offset = 8

    if offset >= len(tokens):
        return board

    if tokens[offset] == 'moves':
        for i in range(offset+1, len(tokens)):
            board.push_uci(tokens[i])

    return board


if len(sys.argv) != 4:
    print("Usage: python3 engine.py <policy> <weights file> <nodes>")
    print(len(sys.argv))
    exit(1)

policy = sys.argv[1]
weights = sys.argv[2]
nodes = int(sys.argv[3])

backend = 'pytorch_cuda' if path.exists('/opt/bin/nvidia-smi') else 'pytorch_cpu'
net = load_network(backend=backend, filename=weights, policy_softmax_temp=2.2)
nn = search.NeuralNet(net=net)

send("Leela Lite")
board = LeelaBoard()

while True:
    line = sys.stdin.readline()
    line = line.rstrip()
    log("<{}".format(line))
    tokens = line.split()
    if len(tokens) == 0:
        continue
    if tokens[0] == "uci":
        send('id name Leela Lite')
        send('id author Dietrich Kappe')
        send('option name List of Syzygy tablebase directories type string default')
        send('uciok')
    elif tokens[0] == "quit":
        exit(0)
    elif tokens[0] == "isready":
        send("readyok")
    elif tokens[0] == "ucinewgame":
        board = LeelaBoard()
    elif tokens[0] == 'position':
        board = process_position(tokens)
    elif tokens[0] == 'go':
        best, node = search.engines[policy](board, nodes, net=nn)
        send("bestmove {}".format(best))
    else:
        print('unknown:', tokens)

# logfile.close()

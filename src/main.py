import os
import time
import importlib.util
import pickle
import sys
from gamestate import GameState

board_len = 12

def lerp(x, a, b, c, d):
    return ((d - c) / (b - a)) * (x - a) + c

def display_grid(gs):
    for r in range(len(gs.grid)):
        for c in range(len(gs.grid)):
            if gs.grid[r][c] == 1:
                print('X', end='')
            elif [r,c] in gs.snake:
                print('S', end='')
            else:
                print('-', end='')
            print(' ',end='')
        print('\n',end='')
    print('\nScore: ' + str(gs.score))
    print('\n\n',end='')

spec = importlib.util.spec_from_file_location("nn", "/Users/alexsmyth/git/neural-network/main.py")
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)
sys.modules['nn'] = nn


def input_func(gs):
    # dist from food, and dist from all 4 walls
    # if move right, up, left, or down
    size = gs.size
    food_pos = gs.get_food_pos()
    def calc_dists(head_pos):
        def dist(r1, c1, r2, c2):
            return ((r2 - r1) ** 2 + (c2 - c1) ** 2) ** 0.5
        return [lerp(dist(head_pos[0], head_pos[1], food_pos[0], food_pos[1]), 0, (2 ** 0.5) * size, 0, 1),
                lerp(head_pos[0], 0, size, 0, 1),
                lerp(size - head_pos[0], 0, size, 0, 1),
                lerp(head_pos[1], 0, size, 0, 1),
                lerp(size - head_pos[1], 0, size, 0, 1)]
    head = gs.snake[0]
    # first 4 inps: light up dir that causes least dist to food
    # next 4 inps: light up dir that is closest to a wall
    # next 4 inps: light up dirs that will lead to hitting snake if move there

    orig_dists = calc_dists([head[0], head[1] + 1]) + calc_dists([head[0] - 1, head[1]])\
        + calc_dists([head[0], head[1] - 1]) + calc_dists([head[0] + 1, head[1]])
    dists = [orig_dists[q*5] for q in range(4)]
    dists = [1 if ele == max(dists) else 0 for ele in dists]
    assert 1 in dists and 0 in dists

    wall_dists = []
    for q in range(4):
        wall_dists.append(orig_dists[q*5+1:(q+1)*5])

    min_dist = 9999999999999999
    closest_to_wall_idx = -1
    for i in range(len(wall_dists)):
        wall_dist = wall_dists[i]
        m = min(wall_dist)
        if m < min_dist:
            min_dist = m
            closest_to_wall_idx = i

    wall_dists = [1 if q == closest_to_wall_idx else 0 for q in range(4)]
    assert 1 in wall_dists and 0 in wall_dists

    snake_dists = [0 for q in range(4)]
    next_heads = [[head[0], head[1] + 1], [head[0] - 1, head[1]], [head[0], head[1] - 1], [head[0] + 1, head[1]]]
    for q in range(len(next_heads)):
        next_head = next_heads[q]
        if next_head in gs.snake:
            snake_dists[q] = 1

    return dists + wall_dists + snake_dists


# Agent, InputFunc -> Final GameState
def run_game_func(agent, input_func):
    gs = GameState(board_len)
    while not gs.is_game_over():
        gs.tick_game(agent, input_func)
    return gs

# Final GameState -> Number
def fitness_func(gs):
    return gs.get_score()
print(__file__)
print(os.path.dirname(__file__))
with open(os.path.join(os.path.dirname(__file__), os.pardir, 'resources/best_net.pkl'), 'rb') as f:
    best_net = pickle.load(f)

all_weights = [neuron.weights for layer in best_net.layers[:len(best_net.layers)-1] for neuron in layer.neurons]


# run a game with the best agent
for z in range(5):
    gs = GameState(board_len)
    while not gs.is_game_over():
        gs.tick_game(best_net, input_func)
        display_grid(gs)
        time.sleep(0.1)

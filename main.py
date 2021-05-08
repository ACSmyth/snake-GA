import random
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time
import importlib.util
import pickle
import sys


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

class GameState:
    def __init__(self, size):
        self.grid = [[0 for q in range(size)] for q in range(size)]
        self.size = size
        self.snake = [[size//2, size//2]]
        self.score = 0
        self.add_food()
        self.dir = 'RIGHT'
        self.game_over = False
        self.ticks_without_eating = 0

    def add_food(self):
        r = random.randint(0, self.size-1)
        c = random.randint(0, self.size-1)
        iters = 0
        while [r,c] in self.snake:
            r = random.randint(0, self.size-1)
            c = random.randint(0, self.size-1)
            iters += 1
            if iters > 10000: raise Exception()
        self.grid[r][c] = 1 # 1 is food
        self.food_pos = [r,c]

    def tick_game(self, agent, input_func):
        # calculate dir
        dirs = ['RIGHT', 'UP', 'LEFT', 'DOWN']
        opposite_dirs = ['LEFT', 'DOWN', 'RIGHT', 'UP']

        output = agent.forward_propagate(input_func(self))
        output = [n.val for n in output]
        # manually set output for opposite of prev direction to zero, since its illegal
        output[opposite_dirs.index(self.dir)] = 0

        chosen_idx = output.index(max(output))
        chosen_dir = dirs[chosen_idx]
        self.dir = chosen_dir

        # move the snake
        prev_head = self.snake[0]
        if self.dir == 'RIGHT':
            self.snake.insert(0, [prev_head[0], prev_head[1] + 1])
        elif self.dir == 'UP':
            self.snake.insert(0, [prev_head[0] - 1, prev_head[1]])
        elif self.dir == 'LEFT':
            self.snake.insert(0, [prev_head[0], prev_head[1] - 1])
        elif self.dir == 'DOWN':
            self.snake.insert(0, [prev_head[0] + 1, prev_head[1]])
        else:
            raise Exception()

        def in_bounds(head):
            return head[0] >= 0 and head[0] < self.size and head[1] >= 0 and head[1] < self.size

        def snake_hit_self(head):
            return head in self.snake[1:len(self.snake)-1]

        def snake_died(head):
            return (not in_bounds(head)) or snake_hit_self(head) or self.ticks_without_eating > self.size**2 // 1.5

        # delete last element, unless there was food
        new_head = self.snake[0]
        if snake_died(new_head):
            self.game_over = True
            return

        if self.grid[new_head[0]][new_head[1]] != 1:
            del self.snake[-1]
        else:
            self.grid[new_head[0]][new_head[1]] = 0
            self.add_food()
            self.score += 1
            self.ticks_without_eating = 0

        self.ticks_without_eating += 1

    def is_game_over(self):
        return self.game_over

    def get_score(self):
        return self.score

    def get_food_pos(self):
        return self.food_pos

'''
# GameState -> NN input (20 dimensions)
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
    return calc_dists([head[0], head[1] + 1]) + calc_dists([head[0] - 1, head[1]])\
        + calc_dists([head[0], head[1] - 1]) + calc_dists([head[0] + 1, head[1]])
'''

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
            snake_dists[q] = 1 # 0 by default, 1 if will hit self

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


'''
mut_rate = 2
gen = nn.GeneticAlgorithm([12, 12, 4], 50, 10, mut_rate, input_func, run_game_func, fitness_func)

best_fit = 0
runs = 1000
for q in range(runs):
    gen.run_generation()
    print(str(q+1) + '/' + str(runs))
    best_agent = gen.get_best_agent()
    fit = best_agent[1]
    best_net = best_agent[0]

    # write best net to file
    if fit > best_fit:
        best_fit = fit
        with open('best_net.pkl', 'wb') as f:
            pickle.dump(best_net, f, pickle.HIGHEST_PROTOCOL)
        print('new best fitness')

    print('fitness: ' + str(fit))
    if fit > 7:
        gen.mut_rate = 1
    if fit > 10:
        gen.mut_rate = 0.25
    if fit > 12:
        gen.mut_rate = 0.05

best_agent = gen.get_best_agent()
best_net = best_agent[0]
#print('avg fitness: ' + str(best_agent[1]))
'''
with open('best_net.pkl', 'rb') as f:
    best_net = pickle.load(f)

all_weights = [neuron.weights for layer in best_net.layers[:len(best_net.layers)-1] for neuron in layer.neurons]
print(all_weights)

# run a game with the best agent
for z in range(5):
    gs = GameState(board_len)
    while not gs.is_game_over():
        gs.tick_game(best_net, input_func)
        display_grid(gs)
        time.sleep(0.1)










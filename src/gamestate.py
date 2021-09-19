import random

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

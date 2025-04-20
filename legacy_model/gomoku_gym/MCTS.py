import math
import copy
import random
import tqdm

C = 0.1 # exploration factor
        # refactor later

class Node:
    def __init__(self, env, player, parent):
        self.env = env
        self.env.render = lambda *args, **kwargs: None
        self.env.unwrapped.render = lambda: None
        self.player = player
        self.parent = parent
        self.children = {}
        self.N = 0
        self.w1 = 0 # num of wins for p1
        self.w2 = 0 # nums of wins for p2
        self.remaining_moves = self.env.unwrapped._get_valid_moves()


    def UCB1(self):
        w = self.w1 if self.player == 1 else self.w2
        return w/(self.N + 1e-3) + C*math.sqrt(2*math.log(self.parent.N)/self.N)
    
    def is_terminal(self):
        p1_win = self.env.unwrapped._win(self.env.unwrapped._p1)
        p2_win = self.env.unwrapped._win(self.env.unwrapped._p2)
        no_moves = len(self.env.unwrapped._get_valid_moves()) == 0
        return p1_win or p2_win or no_moves
    
    def is_fully_expanded(self):
        return len(self.remaining_moves) == 0
    
    def best_child(self):
        best_score = float('-inf')
        best_node = None
        for move, child_node in self.children.items():
            score = child_node.UCB1()
            if score > best_score:
                best_score = score
                best_node = child_node
        return best_node

    

class MCTS:
    def __init__(self, env, player):
        self.env = env
        self.root = Node(copy.deepcopy(env), player, None)

    def tree_descent(self, node):
        current_node = node
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return self.expand(current_node)
            else:
                current_node = current_node.best_child()
        return current_node
    
    def expand(self, node):
        valid_moves = node.remaining_moves
        move = random.choice(valid_moves)
        node.remaining_moves.remove(move)
        new_env = copy.deepcopy(node.env)
        new_env.unwrapped._render_frame = lambda: None
        new_env.step(move)
        child_node = Node(new_env, 3 - node.player, node)
        node.children[move] = child_node
        return child_node

    def rollout(self, env):
        current_env = copy.deepcopy(env)
        current_env.unwrapped._render_frame = lambda: None
        

        while True:
            if current_env.unwrapped._win(current_env.unwrapped._p1):
                return [1, 0]
            if current_env.unwrapped._win(current_env.unwrapped._p2):
                return [0, 1]
            valid_moves = current_env.unwrapped._get_valid_moves()
            if not valid_moves:
                return [0, 0] # draw
            move = random.choice(valid_moves)
            obs, reward, terminated, truncated, info = current_env.step(move)

    def backprop(self, node, reward):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.w1 += reward[0]
            current_node.w2 += reward[1]
            current_node = current_node.parent

    def search(self, num_simulations=1000):
        print("Thinking...")
        pbar = tqdm.tqdm(range(num_simulations))

        for _ in pbar:
            leaf = self.tree_descent(self.root)
            reward = self.rollout(leaf.env)
            self.backprop(leaf, reward)

        best_move = max(self.root.children.items(), key = lambda item: item[1].N)[0]
        return best_move

    def move(self, move, new_env):
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            self.root = Node(copy.deepcopy(new_env), 3 - self.root.player, None)
        


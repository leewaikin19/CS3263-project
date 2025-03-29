import math
import copy
import random
import numpy as np
import torch
from networks import GomokuNet, board_to_tensor

count = 0
savings = 0
class NetworkNode:
    def __init__(self, env, _p1, _p2, valid_moves, player, parent, network):
        self.env = env # Note: this env does NOT contain the correct state. Use _p1 and _p2
        self._p1 = _p1
        self._p2 = _p2
        self.valid_moves = valid_moves
        self.player = player
        self.parent = parent
        self.network = network
        self.children = {}
        self.N = 0
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.P = None  # Prior probabilities

        # Get neural network predictions
        with torch.no_grad():
            board_tensor = board_to_tensor(env, player)
            log_probs, value = network(board_tensor)
            self.P = torch.exp(log_probs).view(15, 15).numpy()
            self.V = value.item()
        
        #self.valid_moves = self.env.unwrapped._get_valid_moves()
        
    def UCB_score(self):
        if self.N == 0:
            return float('inf')  # Always explore unvisited nodes
        return self.Q + 1.0 * math.sqrt(math.log(self.parent.N) / (1 + self.N))
    
    def is_terminal(self):
        p1_win = self.env.unwrapped._win(self._p1)
        p2_win = self.env.unwrapped._win(self._p2)
        no_moves = len(self.valid_moves) == 0
        return p1_win or p2_win or no_moves
    
    def best_child(self):
        return max(self.children.values(), key=lambda child: child.UCB_score())
    
    def most_visited_child(self):
        ## perhaps introduce randomness for max value child?
        delta = 0.3

        if random.random() < delta:
            lst = [child.N for child in self.children.values()]
            max_val = max(lst)  # Find the maximum value
            max_elements = [child for child in self.children.values() if child.N == max_val]  
            return random.choice(max_elements)
        else: 
            return max(self.children.values(), key=lambda child: child.N)

class NetworkMCTS:
    def __init__(self, env, player, network):
        self.env = env
        self.network = network
        self.root = NetworkNode(
            env, 
            copy.deepcopy(env.unwrapped._p1),
            copy.deepcopy(env.unwrapped._p2),
            env.unwrapped._get_valid_moves(),
            player, None, network)

    def search(self, num_simulations=800):
        self.count = 0
        for _ in range(num_simulations):
            node = self._select()
            value = self._evaluate(node)
            self._backpropagate(node, value)
        
        best_child = self.root.most_visited_child()
        #print("clones", self.count)

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)
    
    def _select(self):
        current = self.root
        while current.children:
            current = current.best_child()
        
        if not current.is_terminal() and current.N > 0:
            self._expand(current)
            current = current.best_child()
        
        return current
    
    def _expand(self, node):
        valid_moves = node.valid_moves
        total_p = sum(node.P[y, x] for (x, y) in valid_moves)
        
        for move in valid_moves:
            x, y = move
            #new_env = node.env.unwrapped.clone()
            #self.count+=1
            node.env.step(np.array(move))
            child = NetworkNode(
                node.env, 
                copy.deepcopy(node.env.unwrapped._p1),
                copy.deepcopy(node.env.unwrapped._p2),
                node.env.unwrapped._get_valid_moves(),
                3 - node.player, node, self.network)
            node.env.unwrapped.undo()
            node.children[move] = child
    
    def _evaluate(self, node):
        if node.is_terminal():
            if self.env.unwrapped._win(node._p1):
                return 1 if node.player == 1 else -1
            elif self.env.unwrapped._win(node._p2):
                return 1 if node.player == 2 else -1
            else:
                return 0
        return node.V
    
    def _backpropagate(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value if node.player == 1 else -value
            node.Q = node.W / node.N
            node = node.parent
    
    def move(self, move, new_env):
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            new_child = NetworkNode(copy.deepcopy(new_env), 
                                3 - self.root.player, 
                                None, 
                                self.network)
            self.root.children[move] = new_child  # Add new child to the tree
            self.root = new_child
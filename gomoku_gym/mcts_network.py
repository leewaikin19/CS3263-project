import math
import copy
import random
import numpy as np
import torch
from networks import GomokuNet, board_to_tensor
import concurrent.futures
import threading

count = 0
savings = 0
class NetworkNode:
    def __init__(self, env, _p1, _p2, player, parent, network):
        self.env = env # Note: this env does NOT contain the correct state. Use _p1 and _p2
        self._p1 = _p1
        self._p2 = _p2
        self.player = player
        self.parent = parent
        self.network = network
        self.children = {}
        self.N = 0
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.P = None  # Prior probabilities
        self.lock = threading.Lock()

        # Get neural network predictions
        with torch.no_grad():
            board_tensor = board_to_tensor(env, player)
            log_probs, value = network(board_tensor)
            self.P = torch.exp(log_probs).view(15, 15).numpy()
            self.V = value.item()
        
        self.valid_moves = self.env.unwrapped._get_valid_moves(_p1, _p2)
        
    def UCB_score(self):
        c = 1.0
        if self.N == 0:
            return float('inf')  # Always explore unvisited nodes
        return self.Q + c * math.sqrt(math.log(self.parent.N) / (1 + self.N))
    
    def is_terminal(self):
        p1_win = self.env.unwrapped._win(self._p1)
        p2_win = self.env.unwrapped._win(self._p2)
        no_moves = len(self.valid_moves) == 0
        return p1_win or p2_win or no_moves
    
    def best_child(self):
        ## perhaps introduce randomness for max value child?
        delta = 0.8
        with self.lock:
            if random.random() < delta:
                lst = [child.UCB_score() for child in self.children.values()]
                max_val = max(lst)  # Find the maximum value
                max_elements = [child for child in self.children.values() if child.UCB_score() == max_val]  
                return random.choice(max_elements)
            return max(self.children.values(), key=lambda child: child.UCB_score())
    
    def most_visited_child(self):
        ## perhaps introduce randomness for max value child?
        delta = 0.8
        with self.lock:
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
            player, None, network)

    def search(self, num_simulations=800):
        self.count = 0
        # if num_simulations < moves remaining (225), will not reach terminal
        for _ in range(num_simulations):
            # node is either terminal, unexplored leaf, or child of explored leaf
            node = self._select()
            
            # replace rollout with evaluation
            value = self._evaluate(node)
            # update values
            self._backpropagate(node, value)

        # if self.count > 300:
        #     print("deepcopy", self.count)
        # print(len(self.root.children))
        # if len(self.root.children) == 0:
        #     print("smt is wrong")
        #     print(self.root.is_terminal())
        #     print(self.root.env.unwrapped._win(self.root._p1))
        #     print(self.root.env.unwrapped._win(self.root._p2))
        #     self.env.unwrapped._render_frame(self.root._p1, self.root._p2, self.root.player)
        #     print(len(self.root.valid_moves))
        
            
        best_child = self.root.most_visited_child()
        

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)
    
    def search_parallel(self, num_simulations=800):
        self.count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run the simulations in parallel using map
            futures = [executor.submit(self._iter_parallel) for _ in range(num_simulations)]
            print("Active threads:", threading.active_count())
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Get the result (this could be used for logging or debugging)
                result = future.result()

        best_child = self.root.most_visited_child()

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)

    def _iter_parallel(self):
        # node is either terminal, unexplored leaf, or child of explored leaf
        node = self._select()
        
        # replace rollout with evaluation
        value = self._evaluate(node)

        # update values
        self._backpropagate(node, value)

    def _select(self):
        current = self.root
        ## for convention, use leaf for MCTS graph leaf
        ## use terminal to mean game state ended

        # propagates to leaf or terminal with best_child
        while current.children:
            current = current.best_child()
        
        # if an explored leaf node, return one of its children
        if not current.is_terminal() and current.N > 0:
            self._expand(current)
            current = current.best_child()

        # returns terminal, unexplored leaf, or child of explored leaf 
        return current
    
    # Create and add children nodes
    def _expand(self, node):
        valid_moves = node.valid_moves
        total_p = sum(node.P[y, x] for (x, y) in valid_moves)
        for move in valid_moves:
            x, y = move
            self.count+=1
            orip1 = node._p1
            orip2 = node._p2
            
            if node.player == 1:
                new_p1 = node.env.unwrapped.sim_step(copy.deepcopy(node._p1), node._p2, node.player, np.array(move))
                child = NetworkNode(
                    node.env, 
                    new_p1,
                    node._p2,
                    3 - node.player, node, self.network)
            else:
                new_p2 = node.env.unwrapped.sim_step(node._p1, copy.deepcopy(node._p2), node.player, np.array(move))
                child = NetworkNode(
                    node.env, 
                    node._p1,
                    new_p2,
                    3 - node.player, node, self.network)
            # if np.any(self.root._p1 != self.orip1):
            #     print("PPPPPP33")
            # if np.any(self.root._p2 != self.orip2):
            #     print("QQQQQQQ33")
            # if np.any(node._p1 != orip1) or np.any(node._p2 != orip2):
            #     print("!!!!!!!")
            #     raise ValueError()
            with node.lock:
                node.children[move] = child
    
    def _evaluate(self, node):
        # correct eval if terminal
        if node.is_terminal():
            if self.env.unwrapped._win(node._p1):
                # should be root player val
                # if root is p1 and p1 wins, return +
                return 1 if self.root.player == 1 else -1
            elif self.env.unwrapped._win(node._p2):
                return 1 if self.root.player == 2 else -1
            else:
                return 0
        
        # slightly better evaluation for almost-win (one-move win)
        # if to-play == root.player and almost_win, return win heuristic
        if node.env.unwrapped.almost_win(node._p1) and self.root.player == node.player:
            return 1 if node.player == 1 else -1

        if node.env.unwrapped.almost_win(node._p2) and self.root.player == node.player:
            return 1 if node.player == 2 else -1
        
        # iffy eval if unexplored leaf, or child of explored leaf 
        # node.V should always be in the perspective of p1, 
        # so flip sign when root player is p2
        return node.V if self.root.player == 1 else -node.V
    
    def _backpropagate(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value if node.player == 1 else -value
            node.Q = node.W / node.N
            node = node.parent
    
    def move(self, move, env):
        if move in self.root.children:
            with self.root.lock:
                self.root = self.root.children[move]
            self.root._p1 = env.unwrapped._p1
            self.root._p2 = env.unwrapped._p2
        else:
            self.root = NetworkNode(env, 
                                copy.deepcopy(env.unwrapped._p1),
                                copy.deepcopy(env.unwrapped._p2),
                                3 - self.root.player, 
                                None, 
                                self.network)
import math
import copy
import random
import numpy as np
import torch
from gomoku_gym.networks import GomokuNet
import torch.multiprocessing as mp
from itertools import islice


class NetworkNode:
    def __init__(self, env, _p1, _p2, player, parent, P, V):
        self.env = env # Note: this env does NOT contain the correct state. Use _p1 and _p2
        self._p1 = _p1
        self._p2 = _p2
        self.player = player
        self.parent = parent
        self.children = {}
        self.best_child_node = None
        self.most_visited_child_node = None
        self.N = 0
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.P = P  # Prior probabilities
        self.V = V  # Neural network values

        self.valid_moves = self.env.unwrapped._get_valid_moves(_p1, _p2)

    def UCB_score(self, pol):
        # include policy value
        c = 0.8
        if self.N == 0:
            return float('inf')  # Always explore unvisited nodes
        return self.Q + c * pol * math.sqrt(math.log(self.parent.N) / (1 + self.N))

    def is_terminal(self):
        p1_win = self.env.unwrapped._win(self._p1)
        p2_win = self.env.unwrapped._win(self._p2)
        no_moves = len(self.valid_moves) == 0
        return p1_win or p2_win or no_moves

    def best_child(self):
        ## perhaps introduce randomness for max value child?
        delta = 0.3
        children = self.children.items()
        if random.random() < delta:
            lst = [child.UCB_score(self.P[move]) for move, child in children]
            max_val = max(lst)  # Find the maximum value
            max_elements = [child for move, child in children if child.UCB_score(self.P[move]) == max_val]
            return random.choice(max_elements)

        max_elem = max(children, key=lambda args: args[1].UCB_score(self.P[args[0]]))
        return max_elem[1]

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
    def __init__(self, env, player, network, board_size):
        self.env = env
        self.board_size = board_size
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.V = {} # seems like mcts does not revisit states, no point in hashing
        #self.P = {}
        #hsh = env.unwrapped.hash(env.unwrapped._p1, env.unwrapped._p2, player)
        with torch.no_grad(): # Get neural network predictions
            board_tensor = self.network.board_to_tensor(env.unwrapped._p1, env.unwrapped._p2, player).to(self.device)
            log_probs, value = network(board_tensor)
            P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
            V = value.item()
            #self.P[hsh] = P
            #self.V[hsh] = V
        self.root = NetworkNode(
            env,
            copy.deepcopy(env.unwrapped._p1),
            copy.deepcopy(env.unwrapped._p2),
            player, None, P, V)
        #self.nodes = {}
        #self.nodes[env.unwrapped.hash(env.unwrapped._p1, env.unwrapped._p2, player)] = self.root
        self.num_proc = 8
        self.cuda_pool = CUDAWorkerPool(self.num_proc)

    def search(self, num_simulations=800):
        self.count = 0
        # self.hashsavings = 0
        # if num_simulations < moves remaining (225), will not reach terminal
        for _ in range(num_simulations):
            # node is either terminal, unexplored leaf, or child of explored leaf
            node = self._select()

            # replace rollout with evaluation
            value = self._evaluate(node)
            # update values
            self._backpropagate(node, value)

        # if self.count > 300:
        #print(self.count, "Nodes expanded", "     Hash savings", self.assertt, self.hashsavings, len(self.P))
        print(self.count, "Nodes expanded")
        best_child = self.root.most_visited_child()
        # bug? behaviour where 1 player takes 0.5s but other player takes 30+s
        # is because the best_child chosen from the previous move was not expanded at all from self._select?

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)
        
    def search_parallel(self, num_simulations=800):
        self.count = 0
        # self.hashsavings = 0
        # if num_simulations < moves remaining (225), will not reach terminal
        for _ in range(num_simulations):
            # node is either terminal, unexplored leaf, or child of explored leaf
            node = self._select_parallel_new()

            # replace rollout with evaluation
            value = self._evaluate(node)
            # update values
            self._backpropagate(node, value)

        # if self.count > 300:
        #print(self.count, "Nodes expanded", "     Hash savings", self.assertt, self.hashsavings, len(self.P))
        print(self.count, "Nodes expanded")
        best_child = self.root.most_visited_child()
        # bug? behaviour where 1 player takes 0.5s but other player takes 30+s
        # is because the best_child chosen from the previous move was not expanded at all from self._select?

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)
    
    def search_parallel_old(self, num_simulations=800):
        self.count = 0
        for _ in range(num_simulations):

            selected_nodes = {}
            for i in range(self.num_proc):
                current = self.root
                # propagates to leaf or terminal with best_child
                while current.children:
                    current = current.best_child()
                current.Q = -float('inf') # very dangerous but quick fix to make sure node is never selected
                hsh = self.env.unwrapped.hash(current._p1, current._p2, current.player)
                selected_nodes[hsh] = current

            # Run the simulations in parallel
            #print("len11", len(selected_nodes), self.num_proc)
            results = self.cuda_pool.map(self._iter_parallel, list(selected_nodes.values()))

            for node, parent, value, count in results:
                # update values
                self.count += count
                hsh = self.env.unwrapped.hash(parent._p1, parent._p2, parent.player)
                selected_nodes[hsh].children = parent.children
                self._backpropagate(node, value)


        print(self.count, "Nodes expanded")

        best_child = self.root.most_visited_child()

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)

    def _iter_parallel(self, current):
        # node is either terminal, unexplored leaf, or child of explored leaf
        node, parent, count = self._select_parallel(current)

        # replace rollout with evaluation
        value = self._evaluate(node)

        return node, parent, value, count

    def _select_parallel(self, current):
        # if an explored leaf node, return one of its children
        parent = current
        count = 0
        if not current.is_terminal() and current.N > 0:
            count = self._expand(current)
            current = current.best_child()

        # returns terminal, unexplored leaf, or child of explored leaf
        return current, parent, count

    def _select(self):
        current = self.root
        ## for convention, use leaf for MCTS graph leaf
        ## use terminal to mean game state ended

        # propagates to leaf or terminal with best_child
        while current.children:
            current = current.best_child()

        # if an explored leaf node, return one of its children
        if not current.is_terminal() and current.N > 0:
            self.count += self._expand(current)
            current = current.best_child()

        # returns terminal, unexplored leaf, or child of explored leaf
        return current
    def _select_parallel_new(self):
        current = self.root
        ## for convention, use leaf for MCTS graph leaf
        ## use terminal to mean game state ended

        # propagates to leaf or terminal with best_child
        while current.children:
            current = current.best_child()

        # if an explored leaf node, return one of its children
        if not current.is_terminal() and current.N > 0:
            self.count += self._expand_parallel(current)
            current = current.best_child()

        # returns terminal, unexplored leaf, or child of explored leaf
        return current
    # Create and add children nodes
    def _expand(self, node):
        count = 0
        valid_moves = node.valid_moves
        total_p = sum(node.P[y, x] for (x, y) in valid_moves)
        for m in valid_moves:
            x, y = m
            move = (x, y)

            if node.player == 1:
                new_p1 = node.env.unwrapped.sim_step(copy.deepcopy(node._p1), node._p2, node.player, m)
                with torch.no_grad():
                    board_tensor = self.network.board_to_tensor(new_p1, node._p2, 3 - node.player).to(self.device)
                    log_probs, value = self.network(board_tensor)
                    P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
                    V = value.item()
                count+=1
                child = NetworkNode(
                    node.env,
                    new_p1,
                    node._p2,
                    3 - node.player, node, P, V)
            else:
                new_p2 = node.env.unwrapped.sim_step(node._p1, copy.deepcopy(node._p2), node.player, m)
                # Get neural network predictions
                with torch.no_grad():
                    board_tensor = self.network.board_to_tensor(node._p1, new_p2, 3 - node.player).to(self.device)
                    log_probs, value = self.network(board_tensor)
                    P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
                    V = value.item()
                count+=1
                child = NetworkNode(
                    node.env,
                    node._p1,
                    new_p2,
                    3 - node.player, node, P, V)
            node.children[move] = child
        return count

    # Create and add children nodes
    def _expand_parallel(self, node):
        count = 0
        valid_moves = node.valid_moves
        total_p = sum(node.P[y, x] for (x, y) in valid_moves)
        to_run = []
        for m in valid_moves:
            x, y = m
            move = (x, y)

            if node.player == 1:
                new_p1 = node.env.unwrapped.sim_step(copy.deepcopy(node._p1), node._p2, node.player, m)
                board_tensor = self.network.board_to_tensor(new_p1, node._p2, 3 - node.player).to(self.device)
                to_run.append((move, new_p1, node._p2, board_tensor))
            else:
                new_p2 = node.env.unwrapped.sim_step(node._p1, copy.deepcopy(node._p2), node.player, m)
                board_tensor = self.network.board_to_tensor(node._p1, new_p2, 3 - node.player).to(self.device)
                to_run.append((move, node._p1, new_p2, board_tensor))

        results = self.cuda_pool.map(self._expand_torch, to_run)

        for move, _p1, _p2, log_probs, value in results:
            count+=1
            P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
            V = value.item()
            child = NetworkNode(
                    node.env,
                    _p1,
                    _p2,
                    3 - node.player, node, P, V)
            node.children[move] = child
        return count
    
    def _expand_torch(self, arg):
        m, _p1, _p2, board_tensor = arg
        with torch.no_grad():
            log_probs, value = self.network(board_tensor)
        return (m, _p1, _p2, log_probs, value)
        
    def _batch(self, it):
        while batch := list(islice(it, self.num_proc)):
            yield batch

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
        if node.env.unwrapped.almost_win(node._p1) and node.player == 1:
            return 1 if self.root.player == 1 else -1

        if node.env.unwrapped.almost_win(node._p2) and node.player == 2:
            return 1 if self.root.player == 2 else -1

        # iffy eval if unexplored leaf, or child of explored leaf
        # node.V should always be in the perspective of p1,
        # so flip sign when root player is p2
        return node.V if self.root.player == 1 else -node.V

    def _backpropagate(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def move(self, move, env):
        if move in self.root.children:
            #with self.root.lock:
            self.root = self.root.children[move]
            self.root._p1 = env.unwrapped._p1
            self.root._p2 = env.unwrapped._p2
        else:
            #hsh = env.unwrapped.hash(env.unwrapped._p1, env.unwrapped._p2, 3 - self.root.player)
            #if hsh in self.P:
            #    P = self.P[hsh]
            #    V = self.V[hsh]
            #    self.hashsavings += 1
            #else:
            with torch.no_grad(): # Get neural network predictions
                board_tensor = self.network.board_to_tensor(env.unwrapped._p1, env.unwrapped._p2, 3 - self.root.player).to(self.device)
                log_probs, value = self.network(board_tensor)
                P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
                V = value.item()
                #self.P[hsh] = P
                #self.V[hsh] = V
            self.root = NetworkNode(env,
                                copy.deepcopy(env.unwrapped._p1),
                                copy.deepcopy(env.unwrapped._p2),
                                3 - self.root.player,
                                None,
                                P, V)

class CUDAWorkerPool:
    def __init__(self, num_streams):
        """Initialize a pool of CUDA workers."""
        self.device = torch.device("cuda:0")
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]
        torch.cuda.set_device(self.device)  # Ensure the process uses the correct GPU

    def run_task(self, func, arg, stream_id):
        """Run a function on the worker's CUDA device and return the result."""
        #print("running task")
        with torch.cuda.stream(self.streams[stream_id % self.num_streams]):
            return func(arg)
        
    def map(self, func, tasks):
        """Distribute tasks across CUDA streams asynchronously."""
        results = [self.run_task(func, tasks[i], i) for i in range(len(tasks))]
        torch.cuda.synchronize()  # Ensure all tasks complete
        return results

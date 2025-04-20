from itertools import islice
import math
import copy
import random
import numpy as np
import torch
from gomoku_gym.networks import GomokuNet
import torch.multiprocessing as mp

# Optimisation Idea: Duplicate states with hashmap
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
        self.V = V  # Neural network values from the objective perspective (+1 for P1 win)

        self.valid_moves = self.env.unwrapped._get_valid_moves(_p1, _p2)

    def UCB_score(self, pol, num_moves, board_size, sim_num, num_sims):
        # deviate from AGZ and use a dynamic c instead???
        # c is large at first and goes to 1 as iteration increases
        # c = 4 / (1 + np.exp(10 * (sim_num - num_sims / 2) / num_sims)) + 1
        c = 1 # 5 seems to make the optimal given good policy and values
        # higher c means favour more exploration, use when training to get wide sample data
        # smaller c means favour values more

        #if self.Q + c * pol * math.sqrt(self.parent.N / (1 + self.N)) > 1:
        #    print(self.Q, self.N, c, pol, self.parent.N)

        return self.Q + c * pol * math.sqrt(self.parent.N / (1 + self.N))

    def is_terminal(self):
        p1_win = self.env.unwrapped._win(self._p1)
        p2_win = self.env.unwrapped._win(self._p2)
        no_moves = len(self.valid_moves) == 0
        return p1_win or p2_win or no_moves

    def best_child(self, num_moves, board_size, sim_num, num_sims):
        ## perhaps introduce randomness for max value child?
        delta = 1
        children = self.children.items()
        if random.random() < delta:
            max_elements = []
            max_val = float("-inf")
            for move, child in children:
                score = child.UCB_score(self.P[move], num_moves, board_size, sim_num, num_sims)
                if score > max_val:
                    max_val = score
                    max_elements = [child]
                elif score == max_val:
                    max_elements.append(child)
            lst = np.array([round(child.UCB_score(self.P[move], num_moves, board_size, sim_num, num_sims), 2) for move, child in children])
            # if np.any(lst > 0.9):
            #if len(children) == 16 - num_moves and sim_num % 10 == 0:
                #print("c ", 4 / (1 + np.exp(10 * (sim_num - num_sims / 2) / num_sims)) + 1, "sim num", sim_num, "parent N", self.N)
                #print("N  ", [(move, child.N) for move, child in children])
                #print("Q  ", [(move, round(child.Q, 2)) for move, child in children])
                #print("P  ", [(move, round(self.P[move],2)) for move, child in children])
                #print("V  ", [(move, round(child.V, 2)) for move, child in children])
                #print("UCB", [(move, round(child.UCB_score(self.P[move], num_moves, board_size, sim_num, num_sims), 2)) for move, child in children])
            # self.env.unwrapped._render_frame(self._p1, self._p2, self.player)
            choice =  random.choice(max_elements)
            return choice
            #lst = [child.UCB_score(self.P[move]) for move, child in children]
            #max_val = max(lst)  # Find the maximum value
            #max_elements = [child for move, child in children if child.UCB_score(self.P[move]) == max_val]
            #return random.choice(max_elements)

        max_elem = max(children, key=lambda args: args[1].UCB_score(self.P[args[0]], num_moves, board_size, sim_num, num_sims))
        return max_elem[1]


    def most_visited_child(self, move_num):
        tau = 0
        if move_num == 1:
            tau = 100
        elif move_num == 2:
            tau = 10
        ## perhaps introduce randomness for max value child?
        children = self.children.items()

        # If using temperature τ (i.e., for exploration)
        visit_counts = np.array([child.N for move, child in children], dtype=np.float64)

        if tau == 0:
            # Greedy move selection: choose the most visited move
            max_visits = np.max(visit_counts)
            max_children = [child for move, child in children if child.N == max_visits]
            return random.choice(max_children)

        # Apply softmax with temperature to convert visit counts to probabilities
        exp_visits = np.exp(visit_counts / tau)
        visit_probs = exp_visits / np.sum(exp_visits)
        #print(iteration, -decay_rate*iteration, tau)

        #print(exp_visits)
        # Select a move based on the probabilities
        #print([(move, child.N) for move, child in children])
        #print(visit_counts)
        #print(visit_probs)
        selected_move_idx = np.random.choice(range(len(visit_probs)), p=visit_probs)
        # print("selected move idx", selected_move_idx)
        selected_move = list(self.children.values())[selected_move_idx]
        return selected_move


class NetworkMCTS:
    def __init__(self, env, player, network, board_size, enable_dirichlet=False):
        self.env = env
        self.board_size = board_size
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad(): # Get neural network predictions
            board_tensor = self.network.board_to_tensor(env.unwrapped._p1, env.unwrapped._p2, player).to(self.device)
            log_probs, value = network(board_tensor)
            P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
            V = value.item()
        self.root = NetworkNode(
            env,
            copy.deepcopy(env.unwrapped._p1),
            copy.deepcopy(env.unwrapped._p2),
            player, None, P, V)
        self.num_proc = 1 # does not work, not paralellised
        self.cuda_pool = CUDAWorkerPool(self.num_proc)
        self.num_moves = 0 # 0-indexed
        self.enable_dirichlet = enable_dirichlet

    def dirichlet(self, node):
        alpha = 0.3
        eps = 0.25

        node.P = node.P * self.env.unwrapped._get_valid_mask(node._p1, node._p2)
        node.P = node.P / node.P.sum()
        #print("Pol pred ", *node.P)
        noise = np.random.dirichlet([alpha] * (self.board_size * self.board_size - self.num_moves))
        i = 0
        for move in self.env.unwrapped._get_valid_moves(node._p1, node._p2):
            x, y = move
            node.P[y][x] = (1 - eps) * node.P[y][x] + eps * noise[i]
            i += 1

        #print("dirichlet", *node.P)

    def search_parallel(self, num_simulations=800):
        self.count = 0
        if self.enable_dirichlet:
            self.dirichlet(self.root)

        # if num_simulations < moves remaining (225), will not reach terminal
        for _ in range(num_simulations):
            # node is either terminal, unexplored leaf, or child of explored leaf
            node = self._select_parallel_new(_, num_simulations)

            # replace rollout with evaluation
            value = self._evaluate(node)
            # update values
            self._backpropagate(node, value)

        # if self.count > 300:
        #print(self.count, "Nodes expanded", "     Hash savings", self.assertt, self.hashsavings, len(self.P))
        print(self.count, "Nodes expanded")
        best_child = self.root.most_visited_child(self.num_moves)
        # bug? behaviour where 1 player takes 0.5s but other player takes 30+s
        # is because the best_child chosen from the previous move was not expanded at all from self._select?

        # Return action as a tuple (x,y)
        return next(move for move, child in self.root.children.items() if child == best_child)

    def _select_parallel_new(self, sim_num, num_sims):
        current = self.root
        ## for convention, use leaf for MCTS graph leaf
        ## use terminal to mean game state ended

        # propagates to leaf or terminal with best_child
        while current.children:
            current = current.best_child(self.num_moves, self.board_size, sim_num, num_sims)

        # if an explored leaf node, return one of its children
        if not current.is_terminal() and current.N > 0:
            self.count += self._expand_parallel(current)
            current = current.best_child(self.num_moves, self.board_size, sim_num, num_sims)

        # returns terminal, unexplored leaf, or child of explored leaf
        return current


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
            V = value.item() # this is the objective view (+1 for P1 win)
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
        # evaluate should return a value from the node's parent's player's perspective
        # instead of the root, because in backpropagate we flip the values
        if node.is_terminal():
            #print("player", 3 - node.player, "wins, root", self.root.player)
            if self.env.unwrapped._win(node._p1):
                # Note it is flipped, because node.player is the player to play
                # But the evaluation is on the previous action made by the previous player
                return 1 if node.player == 2 else -1
            elif self.env.unwrapped._win(node._p2):
                return 1 if node.player == 1 else -1
            else:
                return 0

        return node.V if node.player == 2 else -node.V

    def _backpropagate(self, node, value):
        # start with assigning the value to the node
        # value is retuirned by _evaluate and is from the node's parent's perspective
        #
        # if value == 1 or value == -1:
        #     print("assigning some node", value, "as player", node.player, "with root", self.root.player)
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent

    def move(self, move, env):
        self.num_moves += 1
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root._p1 = env.unwrapped._p1
            self.root._p2 = env.unwrapped._p2
        else:
            with torch.no_grad(): # Get neural network predictions
                board_tensor = self.network.board_to_tensor(env.unwrapped._p1, env.unwrapped._p2, env.unwrapped.player).to(self.device)
                log_probs, value = self.network(board_tensor)
                P = torch.exp(log_probs).view(self.board_size, self.board_size).cpu().numpy()
                V = value.item()
            self.root = NetworkNode(env,
                                copy.deepcopy(env.unwrapped._p1),
                                copy.deepcopy(env.unwrapped._p2),
                                env.unwrapped.player,
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
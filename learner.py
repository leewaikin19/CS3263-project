import torch
import copy
import numpy as np
from search import MCTSParallel
import random
from tqdm import trange
import torch.nn.functional as F


class ParallelLearner:
    def __init__(self, model, game, optimizer, args):
        self.model = model
        self.game = game
        self.optimizer = optimizer
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spgs = [SPG(self.game) for spg in range(self.args["num_parallel_games"])]

        while len(spgs) > 0:
            states = np.stack([spg.state for spg in spgs])

            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spgs)

            for i in range(len(spgs))[::-1]:
                spg = spgs[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                spg.state = self.game.get_next_state(spg.state, action, player)
                next_p1 = copy.deepcopy(spg.p2)
                next_p2 = self.game.step(action, copy.deepcopy(spg.p1))

                value, is_terminal = self.game.get_value_and_terminated(spg.state, next_p2)

                if is_terminal:

                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            	self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome,
                            ))

                    del spgs[i]
                    
                spg.p1, spg.p2 = next_p1, next_p2  # worst bug ever, had this unindented

            player = self.game.get_opponent(player)  # another worst bug ever, had this indented

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        avg_batch_loss = None
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            avg_batch_loss = loss.item() if avg_batch_loss is None else avg_batch_loss * 0.9 + loss.item() * 0.1 #Keep an EWA # Keep an EWA

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return avg_batch_loss

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            avg_loss = None
            memory = []

            self.model.eval()  #crucial! Have to set to eval mode!
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                avg_batch_loss = self.train(memory)
                avg_loss = avg_batch_loss if avg_loss is None else avg_loss * 0.9 + avg_loss * 0.1 #keep another EWA

            print(f"iteration loss is {avg_loss}")
            torch.save(self.model.state_dict(), f'model_{iteration}_{self.game}.pt')
            torch.save(self.optimizer.state_dict(), f'optimizer_{iteration}_{self.game}.pt') #save optimizer, in case you want to retrain


class SPG:
    def __init__(self, game):
        self.state, self.p1, self.p2 = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

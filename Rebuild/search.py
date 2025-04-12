import torch
import copy
import numpy as np
import math

class Node:
    def __init__(self, game, args, state, _p1, _p2, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self._p1 = _p1 #np.array([0] * self.board_size, dtype=np.uint16)
        self._p2 = _p2 #np.array([0] * self.board_size, dtype=np.uint16)
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                next_p1 = copy.deepcopy(self._p2)
                next_p2 = self.game.step(action, copy.deepcopy(self._p1))

                child = Node(self.game, self.args, child_state, next_p1, next_p2, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)




class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state, _p1, _p2):
        root = Node(self.game, self.args, state, _p1, _p2, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node._p2) #unfortunate bug, had node._p1
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    


class MCTSParallel:
  def __init__(self, game, args, model):
    self.game = game
    self.args = args
    self.model = model

  @torch.no_grad()
  def search(self, states, spgs):


    policy, _ = self.model(
        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
    )

    policy = torch.softmax(policy, axis=1).cpu().numpy()
    policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

    for i, spg in enumerate(spgs):
      spg_policy = policy[i]
      # print(spg_policy)
      valid_moves = self.game.get_valid_moves(states[i])
      # print(valid_moves)
      spg_policy *= valid_moves
      spg_policy /= np.sum(spg_policy)


      spg.root = Node(self.game, self.args, states[i], spg.p1, spg.p2, visit_count=1)
      spg.root.expand(spg_policy)

    for search in range(self.args['num_searches']):
      for spg in spgs:
        spg.node = None
        node = spg.root

        while node.is_fully_expanded():
          node = node.select()

        value, is_terminal = self.game.get_value_and_terminated(node.state, node._p2) #unfortunate bug, had node._p1
        value = self.game.get_opponent_value(value)

        if is_terminal:
          node.backpropagate(value)
        else:
          spg.node = node

      expandable_spgs = [mappingIdx for mappingIdx in range(len(spgs)) if spgs[mappingIdx].node is not None]

      if  len(expandable_spgs) > 0:
        states = np.stack([spgs[mappingIdx].node.state for mappingIdx in expandable_spgs])

        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        value = value.cpu().numpy()

      for i, mappingIdx in enumerate(expandable_spgs):
        node = spgs[mappingIdx].node
        spg_policy, spg_value = policy[i], value[i]

        valid_moves = self.game.get_valid_moves(node.state)
        spg_policy *= valid_moves
        spg_policy /= np.sum(spg_policy)
        spg_value = self.game.get_opponent_value(spg_value)

        node.expand(spg_policy)
        node.backpropagate(spg_value)
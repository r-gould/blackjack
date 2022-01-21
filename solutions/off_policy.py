import numpy as np

from .solution import Solution

class OffPolicy(Solution):
    name = "Off-Policy Monte Carlo"

    def __init__(self, state_space, action_space, param_dict={}):
        super().__init__(state_space, action_space, param_dict)
        self.behaviour = {state: [1/len(action_space) for _ in range(len(action_space))] for state in state_space}
        self.cumulative = {state: [0 for _ in range(len(action_space))] for state in state_space}

    def update(self, episode):
        states, actions, rewards = episode
        T = len(states)
        G = 0
        W = 1
        for t in range(T-1, -1, -1):
            reward = rewards[t]
            G = self.gamma * G + reward
            
            state, action = states[t], actions[t]
            player_sum = state[0]
            if player_sum < 12:
                continue
                # Policy for sum < 12 is always to hit, so no need to update policy.

            self.cumulative[state][action] += W
            Q = self.values[state][action]
            C = self.cumulative[state][action]
            self.values[state][action] += W * (G - Q) / C
            
            self.greedy_update(state)
            best_action = np.argmax(self.policy[state])
            if action != best_action:
                break
            W *= 1 / self.behaviour[state][action]

    def greedy_update(self, state):
        best_action = self.value_argmax(state)
        for action in self.action_space:
            if action == best_action:
                self.policy[state][action] = 1
                continue
            self.policy[state][action] = 0

    def episode_policy(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1

        return np.random.choice(self.action_space, p=self.behaviour[state])

    def final_policy(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1

        return np.argmax(self.policy[state])
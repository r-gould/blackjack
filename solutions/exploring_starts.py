import random
import numpy as np

from .solution import Solution

class ExploringStarts(Solution):
    name = "Monte Carlo with Exploring Starts"
    
    def __init__(self, state_space, action_space, param_dict={}):
        super().__init__(state_space, action_space, param_dict)
        self.start = True
        self.every_visit = param_dict.get("every_visit") or False

    def update(self, episode):
        states, actions, rewards = episode
        T = len(states)
        G = 0
        for t in range(T-1, -1, -1):
            reward = rewards[t]
            G = self.gamma * G + reward
            
            state, action = states[t], actions[t]
            player_sum = state[0]
            if player_sum < 12:
                continue
                # Policy for sum < 12 is always to hit, so no need to update policy.
            visited = state in states[:t] and action in actions[:t]
            
            if self.every_visit or not visited:
                self.counts[state][action] += 1
                Q = self.values[state][action]
                N = self.counts[state][action]
                self.values[state][action] += (G - Q) / N
                
                self.greedy_update(state)
                
        self.start = True
    
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

        if self.start:
            self.start = False
            return random.choice(self.action_space)

        return np.argmax(self.policy[state])

    def final_policy(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1

        return np.argmax(self.policy[state])
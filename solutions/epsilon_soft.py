import numpy as np

from .solution import Solution

class EpsilonSoft(Solution):
    name = "Monte Carlo with epsilon-soft policies"

    def __init__(self, state_space, action_space, param_dict):
        super().__init__(state_space, action_space, param_dict)
        self.epsilon = param_dict["epsilon"]
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
                
                self.soft_update(state)

    def soft_update(self, state):
        best_action = self.value_argmax(state)
        e_soft = self.epsilon / len(self.action_space)
        for action in self.action_space:
            if action == best_action:
                self.policy[state][action] = 1 - self.epsilon + e_soft
                continue
            self.policy[state][action] = e_soft
    
    def episode_policy(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1
            
        action = np.random.choice(self.action_space, p=self.policy[state])
        return action

    def final_policy(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1

        return np.argmax(self.policy[state])
import numpy as np

class Solution:
    name = None

    def __init__(self, state_space, action_space, param_dict={}):
        self.state_space = state_space
        self.action_space = action_space

        self.policy = {state: [1/len(action_space) for _ in range(len(action_space))] for state in state_space}
        self.values = {state: [0 for _ in range(len(action_space))] for state in state_space}
        self.counts = {state: [0 for _ in range(len(action_space))] for state in state_space}
        self.gamma = param_dict.get("gamma") or 1

    def value_func(self, state, action):
        return self.values[state][action]
    
    def value_argmax(self, state):
        return np.argmax(self.values[state])

    def update(self, episode):
        raise NotImplementedError()

    def episode_policy(self, state):
        raise NotImplementedError()
    
    def final_policy(self, state):
        raise NotImplementedError()
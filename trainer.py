class Trainer:
    def __init__(self, env):
        self.env = env

    def run(self, solution, num_episodes):
        for _ in range(num_episodes):
            episode = self.run_episode(solution)
            solution.update(episode)
        
        return solution

    def run_episode(self, solution):
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        while True:
            states.append(state)
            action = solution.episode_policy(state)
            actions.append(action)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        
        return states, actions, rewards
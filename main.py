import gym

from trainer import Trainer
from solutions.exploring_starts import ExploringStarts
from solutions.epsilon_soft import EpsilonSoft
from solutions.off_policy import OffPolicy
from utils import plot_solution, policy_table, blackjack_spaces

def main(solutions, num_episodes):
    env = gym.make("Blackjack-v1")
    trainer = Trainer(env)
    for solution in solutions:
        trainer.run(solution, num_episodes)
        table = policy_table(solution)
        plot_solution(solution)

if __name__ == "__main__":
    state_space, action_space = blackjack_spaces()
    solutions = [
        ExploringStarts(state_space, action_space),
        EpsilonSoft(state_space, action_space, {
            "epsilon": 0.1,
        }),
        OffPolicy(state_space, action_space),
    ]
    num_episodes = 1000000
    main(solutions, num_episodes)
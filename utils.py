import numpy as np
import matplotlib.pyplot as plt

def plot_solution(solution):
    player_sums = [i for i in range(12, 22)]
    dealer_cards = [i for i in range(1, 11)]
    values = np.zeros((10, 10, 2))

    for player in player_sums:
        for dealer in dealer_cards:
            for usable in [0, 1]:
                state = (player, dealer, usable)
                action = solution.final_policy(state)
                values[player-12, dealer-1, usable] = solution.value_func(state, action)

    fig, axes = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})
    fig.suptitle(solution.name)
    titles = ["No usable ace", "Usable ace"]
    X, Y = np.meshgrid(player_sums, dealer_cards)

    for i in [0, 1]:
        ax = axes[i]
        ax.plot_wireframe(X, Y, values[:, :, i].T)
        ax.set_xlabel("Player sum")
        ax.set_ylabel("Dealer card")
        ax.set_zlabel("Value")
        ax.set_title(titles[i])

    plt.show()

def policy_table(solution):
    player_sums = [i for i in range(12, 22)]
    dealer_cards = [i for i in range(1, 11)]
    policy_table = np.zeros((10, 10, 2))
    
    for player in player_sums:
        for dealer in dealer_cards:
            for usable in [0, 1]:
                state = (player, dealer, usable)
                policy_table[player-12, dealer-1, usable] = solution.final_policy(state)
    
    return policy_table
    
def blackjack_spaces():
    state_space = []
    for player in range(12, 22):
        for dealer in range(1, 11):
            for usable in [0, 1]:
                state = (player, dealer, usable)
                state_space.append(state)
    
    action_space = [0, 1]
    return state_space, action_space
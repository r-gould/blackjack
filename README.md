# blackjack
An implementation of various solutions for a simplified version of blackjack, as described in 'Reinforcement Learning: An Introduction (Richard S. Sutton, Andrew G. Barto)'

Solutions implemented:
* Monte Carlo with ES (Exploring Starts)
* On-policy first-visit Monte Carlo control (for epsilon-soft policies)
* Off-policy Monte Carlo control

All solutions converge to the optimal policy shown below,

|    || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---||---|---|---|---|---|---|---|---|---|---|
| 12 || H | H | H | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 13 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 14 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 15 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 16 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 17 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 18 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 19 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 20 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 21 || A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |

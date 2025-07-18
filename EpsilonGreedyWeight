import numpy as np
from kArmBandit import kbandit

rng = np.random.default_rng()

actions = 10
ep = 0.1

stationary = True

Q = np.zeros(actions)
N = np.zeros(actions)

print(f'Starting stepsize weight average method with ep = {ep}')

bandit = kbandit(stationary, actions)

for step in range(10000):
    if rng.random() > ep:
        a = Q.argmax()
    else:
        a = rng.integers(low=0, high=(actions-1))

    N[a] += 1

    reward = bandit.kbandit(a)

    Q[a] = Q[a] + (reward - Q[a])* (1/N[a])

bandit.graph(Q)
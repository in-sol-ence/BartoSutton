import numpy as np
import matplotlib.pyplot as plt
from kArmBandit import kbandit

rng = np.random.default_rng()

actions = 10
ep = 0.1
alpha = 0.2
rewcum= 0
stationary = False
Q = np.zeros(actions)
N = np.zeros(actions)
steps = []
rewards=[]
opt=[]
opt.append(0)

print(f'Starting stepsize constant method with ep = {ep} and stepsize const = {alpha}')
bandit = kbandit(stationary, actions)
for step in range(10000):
    if rng.random() > ep:
        a = Q.argmax()
    else:
        a = rng.integers(low=0, high=(actions-1))
    
    reward = bandit.kbandit(a)

    Q[a] = Q[a] + alpha*(reward - Q[a])
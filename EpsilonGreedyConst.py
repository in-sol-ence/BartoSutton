import numpy as np
import matplotlib.pyplot as plt
import "k-arm-bandit"
rng = np.random.default_rng()

actions = 10
ep = 0.1
alpha = 0.2
rewcum=0

Q = np.zeros(actions)
N = np.zeros(actions)
steps = []
rewards=[]
opt=[]
opt.append(0)

print(f'Starting stepsize constant method with ep = {ep} and stepsize const = {alpha}')
bandit = kbandit(stat, actions)
for step in range(10000):
    if rng.random() > ep:
        a = Q.argmax()
    else:
        a = rng.integers(low=0, high=(actions-1))
    
    reward = bandit.kbandit(a)
    rewcum+=reward
    rewards.append(rewcum)
    Q[a] = Q[a] + alpha*(reward - Q[a])
    steps.append(step)
    if len(opt)>0 and step>0:
        i=opt[len(opt)-1]
        if a == bandit.optimal():
            opt.append((i*(step-1) + 100)/step)
        else:
            opt.append(i*(step-1)/step)
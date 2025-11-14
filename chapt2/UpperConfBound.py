import numpy as np
from kArmBandit import kbandit

actions = 10

c = 3 # This is the exploration constant 
const = False # Q values are updated using this. Set as const but can change it to weighted by setting this to false
Q = np.zeros(actions)
N = np.zeros(actions)
alpha = 0.2

bandit = kbandit( stationary=False, actions=actions)

print(f'Starting upper confidence bound action selection with exploration constant = {c}. Constant = {const}')

for steps in range(10000):
    max = Q[0] + c*((np.log(steps)/N[0])**0.5)
    action = 0
    for a in range(actions):
        At = Q[a] + c*((np.log(steps)/N[a])**0.5)
        if At > max:
            action = a
            max = At
        
    reward = bandit.kbandit(action)

    N[action]+=1

    if const == True:
        Q[action] = Q[action] + (reward - Q[action])*alpha
    else:
        Q[action] = Q[action] + (reward - Q[action])* (1/(N[action]))

bandit.graph(Q)
import numpy as np 
from random import randint as r
import random
import matplotlib.pyplot as plt

def run_episode():
    """This returns a trajectory tao and final reward."""
    a = 0
    ret = []
    while not (a <= -3 or a >= 3):
        ret.append(a+2)
        a += 1 if r(1, 10) > 5 else -1
    return ret, 1 if a==3 else 0

alpha = 0.001
v = np.zeros(5)
v.fill(0.5) 
# new: track history for plotting
num_episodes = 10000
v_history = np.zeros((num_episodes+1, v.size))
v_history[0] = v.copy()

random.seed(0)  # reproducible runs

for i in range(num_episodes):
    tao, reward = run_episode()
    for t in tao: 
        v[t] = v[t] + alpha*(reward - v[t])
    v_history[i+1] = v.copy()

print(v)

# plot learning curves for each state
episodes = np.arange(num_episodes+1)
for s in range(v.size):
    plt.plot(episodes, v_history[:, s], label=f'state {s}')
plt.xlabel('Episode')
plt.ylabel('Value estimate')
plt.title('Value estimates over episodes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning.png', dpi=150)
plt.show()
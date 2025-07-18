import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

class kbandit:
    def __init__(self, stationary=True, actions =10, standard_deviation=0.01):
        self.stat = stationary
        self.rewmean = rng.uniform(low=-20, high=10, size=actions) # The randomly generated true values of the means of the reward distributions.
        self.stdv = standard_deviation # Standard deviation of rewards
        self.act = actions
        self.rewcum = 0 # Cumulative reward
        self.step = 0 # Time steps are defined as each time an action is taken.
        self.opt = [] # This is an array that stores for each time step the percentage of the optimal actions taken before that time step. 
        self.rewavg = [] # Not to be confused with rewmean. This is an array that stores for each time step the average of all rewards before that time step. 

    def kbandit(self, action):
        if action>=0 and action<self.act:
            reward = rng.normal(loc=self.rewmean[action], scale=self.stdv)
        else:
            print("Invalid Action") 
            print(f'Action: {action}')

        if (not self.stat) and rng.random()>0.5:
            self.rewmean[rng.integers(low=0, high=(self.act-1))]+=(rng.normal(loc=0, scale=0.2)) #Edit scale (stdev) to change how fast the values change
        self.rewcum+=reward
        self.step+=1

        if len(self.opt) == 0: # For the first time step the 'value at the time step before' is 0.
            prevopt = 0
        else:
            try:
                prevopt = self.opt[len(opt)-1]
            except IndexError:
                print(opt)
        
        
        if a == bandit.optimal():
            self.opt.append((prevopt*(self.step-1) + 100)/self.step)
        else:
            self.opt.append(prevopt*(self.step-1)/self.step)

        if len(self.rewavg) == 0:
            prevrewavg = 0
        else:
            prevrewavg = self.rewavg[len(self.rewavg)-1]
        self.rewavg.append(prevrewavg + ((1/self.step) * (reward - prevrewavg))) # This is another update rule you have seen before. Look at EpsilonGreedyWeighted
        return reward
    
    def cheat(self):
        return self.rewmean
    
    def optimal(self):
        return self.rewmean.argmax()
    
    def graph(self, Q):
        (np.arange(0, step)) 

        print("Action Q vals : ")
        print(Q)
        print("True values :")
        print(self.cheat())
        print(f'Cumulative reward = {rewcum}')

        plt.plot(steps, rewards, marker='o')  # 'o' adds a dot at each step
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Step-rewards Plot')
        plt.grid(True)
        plt.show()

        plt.plot(steps, opt, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Optimal Action %')
        plt.title('Optimal Action Plot')
        plt.grid(True)
        plt.show() 

    
actions = 10
ep = 0.1
alpha = 0.2
rewcum=0

stat=False
s = input("Stationary or non stationary. y or n     ")
if s == "y":
    stat=True

avg=True
c = input("Stepsize constant or weighted average. c or v       ")
if c=="c":
    avg=False

Q = np.zeros(actions)
N = np.zeros(actions)
steps = []
rewards=[]
opt=[]
opt.append(0)

if not avg:
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

else:
    print(f'Starting stepsize weight average method with ep = {ep}')
    bandit = kbandit(stat, actions)
    for step in range(10000):
        if rng.random() > ep:
            a = Q.argmax()
        else:
            a = rng.integers(low=0, high=(actions-1))
        N[a] += 1
        reward = bandit.kbandit(a)
        rewcum+=reward
        rewards.append(rewcum)
        Q[a] = Q[a] + (reward - Q[a])* (1/N[a])
        steps.append(step)
        if len(opt)>0 and step>0:
            i=opt[len(opt)-1]
            if a == bandit.optimal():
                opt.append((i*(step-1) + 100)/step)
            else:
                opt.append(i*(step-1)/step)

bandit.graph(Q)
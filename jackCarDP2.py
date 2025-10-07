import time as time
import matplotlib.pyplot as plt
from types import SimpleNamespace
import math
import numpy as np

def init() -> dict:
    ## Initialize policy and value function
    # Considering it is a deterministic environment, we can use a dictionary to store the policy and value function
    policy = {}
    value = {}
    
    env = SimpleNamespace()
    
    env.r1 = 3 # lambda values for the poisson variable for rentals taken
    env.r2 = 4 
    env.m1 = 3 # lambda values for the poisson variable for rental returns
    env.m2 = 2
    lamdas = [env.r1, env.r2, env.m1, env.m2]
    env.rentRevenue = 10
    env.moveCost = -5
    env.max_cars = 20
    env.max_move = 5
    env.max_poisson = 20 +1 # We will consider poisson values only up to # since the probability of getting a value more than 11 is negligible for our lambda values. +1 for range function.
    for cars1 in range(env.max_cars + 1): # 0 to 20 cars
        for cars2 in range(env.max_cars + 1):
            policy[(cars1, cars2)] = 0  # Initial policy: no cars moved
            value[(cars1, cars2)] = 0.0
    
    def poi(lamda, n):
        """Calculates poisson probability of n given lambda"""
        return (lamda**n)*(math.exp(-lamda))/math.factorial(n)
    
    env.poiMap = dict()
    for l in lamdas:
        env.poiMap[(l, 0)] = poi(l, 0)
        for k in range(1, env.max_poisson):
            env.poiMap[(l, k)] = env.poiMap[(l, k-1)]*l/k # Using the property that P(X=n) = (lambda/n)*P(X=n-1)
    
    print("Initialized environment with poisson map")
    
    return {
        "value": value,
        "policy": policy,
        "env": env
    }

def policyEval(gamma, theta, value, policy, env) -> dict:
    """ Given state-values, a policy, certain environment dynamics and a value theta which defines accuracy of the state-values,
    policyEval calculates new state-value pairs for the given policy to accuracy theta. 
    
    It returns the new state-value pairs"""
    # Policy Evaluation
    delta = theta+1
    policy_eval = 0
    while (delta > theta):
        delta = 0.0   
        for cars1 in range(env.max_cars + 1):
            for cars2 in range(env.max_cars + 1):
                old = value[(cars1, cars2)]
                new = 0
                action = policy[(cars1, cars2)]
                for m1 in range(0, env.max_poisson):
                    for m2 in range(0, env.max_poisson):
                        for r1 in range(0, env.max_poisson):
                            for r2 in range(0, env.max_poisson):
                                if cars1-action < 0:
                                    action = cars1
                                elif action+cars2 < 0:
                                    action = -(cars2)
                                
                                c1 = cars1-action # This is the cars after the action
                                c2 = cars2+action
                                
                                d1 = max(c1-r1, 0)
                                d2 = max(c2-r2, 0)   
                                nextState = (min(d1+m1, env.max_cars), min(d2+m2, env.max_cars)) # This is the next state. 
                                stateTransitonProb = env.poiMap[(env.r1, r1)]*env.poiMap[(env.r2, r2)]*env.poiMap[(env.m1, m1)]*env.poiMap[(env.m2, m2)]
                                new += (stateTransitonProb)*((min(c1, r1)+min(c2, r2))*env.rentRevenue + gamma*value[nextState])
                if action<0: 
                    new += (action-1)*env.moveCost
                else:
                    new += action*env.moveCost
                value[(cars1, cars2)] = new
                delta = max(delta, abs(old-new)) # Recording the max difference in new and old state values. Eventually this should converge to 0 and the state values for the policy pi should conveerge to their values but to save compute we set a tolerance theta.
        policy_eval+=1
        print(f"Policy evaluation {policy_eval} completed with delta: {delta}.")
    return value

def policyImprov(gamma, value, policy, env) -> dict:
    """Given accurate (to theta) state-value pairs and certain environment dynamics,
    policyImprov finds the best policy pi.
    It returns a new policy and a boolean indicating the old policy's stability."""
    policy_stable = True
    for cars1 in range(env.max_cars + 1):
        for cars2 in range(env.max_cars + 1):
            best_action = policy[(cars1, cars2)]
            orig_action = best_action
            argMax = value[(cars1, cars2)]
            for action in range(-5, 6):
                if (action > 0 and cars1 < action) or (action < 0 and cars2 < abs(action)):
                    continue # Invalid action
                if (cars1 - action > env.max_cars) or (cars2 + action > env.max_cars):
                    continue # Invalid action
                actionVal=0
                for m1 in range(0, env.max_poisson):
                    for m2 in range(0, env.max_poisson):
                        for r1 in range(0, env.max_poisson):
                            for r2 in range(0, env.max_poisson):
                                c1 = cars1-action # This is the cars after the action
                                c2 = cars2+action
                                d1 = max(c1-r1, 0)
                                d2 = max(c2-r2, 0)
                                nextState = (min(d1+m1, env.max_cars), min(d2+m2, env.max_cars)) # This is the next state. 
                                stateTransitonProb = env.poiMap[(env.r1, r1)]*env.poiMap[(env.r2, r2)]*env.poiMap[(env.m1, m1)]*env.poiMap[(env.m2, m2)]
                                actionVal += (stateTransitonProb)* ((min(c1, r1)+ min(c2, r2))*env.rentRevenue + gamma*value[nextState])

                if action<0: 
                    actionVal += (action-1)*env.moveCost
                else:
                    actionV += action*env.moveCost
                if argMax < actionVal:
                    best_action = action
                    argMax = actionVal    
                
            if orig_action != best_action:
                print(f'Updated policy state: {cars1},{cars2}. From action: {orig_action} to {best_action}')
                policy_stable = False
                policy[(cars1, cars2)] = best_action
                
    if policy_stable:
        print("Found optimal policy")
        return policy, policy_stable 
    else:
        print("Going back in")
        return policy, policy_stable

def graphHelper(value: dict, figName: str, env=SimpleNamespace(max_cars=20)):
    board = np.zeros((env.max_cars+1, env.max_cars+1), dtype=float)

    for car1 in range(env.max_cars + 1):
        for car2 in range(env.max_cars + 1):
            board[car1, car2] = round(value[(car1, car2)])

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(board, origin="lower", interpolation="none", aspect="equal")

    # Tick marks at each cell
    ax.set_xticks(range(env.max_cars+1))
    ax.set_yticks(range(env.max_cars+1))
    ax.set_xlabel("cars1 (x)")
    ax.set_ylabel("cars2 (y)")

    # Cell values
    for y in range(env.max_cars+1):
        for x in range(env.max_cars+1):
            ax.text(x, y, f"{board[y, x]:g}", ha="center", va="center", fontsize=7)

    # Grid lines on cell boundaries
    ax.set_xticks(np.arange(-.5, env.max_cars+1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.max_cars+1, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, label="value")
    plt.tight_layout()
    plt.savefig(figName)
    print(f'Saved graph at {figName}')

    time.sleep(1)

def main():
    theta = 1  # Learning rate
    gamma = 0.9  # Discount factor
    var = init()
    policy_stable = False
    iter = 0
    while not policy_stable:
        graphHelper(var["value"], f'Values_{iter}_new')
        graphHelper(var["policy"], f'Policy_{iter}_new')
        var["value"] = policyEval(gamma, theta, **var) # updating the state-values
        var["policy"], policy_stable = policyImprov(gamma, **var) # updating the policy
        
        iter+=1
    print(var["policy"])
    graphHelper(var["policy"], f'OptimalPolicy_new')


if __name__ == "__main__":
    main()
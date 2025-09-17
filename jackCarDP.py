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
    for cars1 in range(22):
        for cars2 in range(22):
            policy[(cars1, cars2)] = 0  # Initial policy: no cars moved
            value[(cars1, cars2)] = 0.0
    
    env = SimpleNamespace()
    
    env.r1 = 3
    env.r2 = 4
    env.m1 = 3
    env.m2 = 2
    env.rentRevenue = 10
    env.moveCost = -5
    env.max_cars = 20
    env.max_move = 5
    env.move_cost = 2
    
    def poissonCalc(lambda_, k): 
        return (lambda_**k * math.exp(-lambda_))/math.factorial(k)
    
    env.poisson = poissonCalc
    
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
        for cars1 in range(22):
            for cars2 in range(22):
                old = value[(cars1, cars2)]
                new = 0
                action = policy[(cars1, cars2)]
                for m1 in range(0, cars1):
                    for m2 in range(0, cars2):
                        for r1 in range(0, cars1):
                            for r2 in range(0, cars2):
                                c1 = min(max(cars1-action, 0), 21) # This is the cars after the action
                                c2 = min(max(cars2+action, 0), 21 )
                                nextState = (min(max(c1-r1+m1, 0), 21), min(max(c1-r1+m1, 0), 21)) # This is the next state. 
                                stateTransitonProb = env.poisson(env.r1, r1)*env.poisson(env.r2, r2)*env.poisson(env.m1, m1)*env.poisson(env.m2, m2)
                                new += (stateTransitonProb)* ((c1+c2)*env.rentRevenue + gamma*value[nextState])
                new += action*env.moveCost
                value[(cars1, cars2)] = new
                delta = max(delta, abs(old-new)) # Reecording the max difference in new and old state values. Eventually this should converge to 0 and the state values for the policy pi should conveerge to their values but to save compute we set a tolerance theta.
        policy_eval+=1
        print(f"Policy evaluation {policy_eval} completed with delta: {delta}.")
    return value

def policyImprov(gamma, value, policy, env) -> dict:
    """Given accurate (to theta) state-value pairs and certain environment dynamics,
    policyImprov finds the best policy pi.
    
    It returns a new policy and a boolean indicating the old policy's stability."""
    policy_stable = True
    for cars1 in range(22):
        for cars2 in range(22):
            best_action = policy[(cars1, cars2)]
            orig_action = best_action
            argMax = value[(cars1, cars2)]
            actionVal=0
            for action in range(-5, 6):
                for m1 in range(0, cars1):
                    for m2 in range(0, cars2):
                        for r1 in range(0, cars1):
                            for r2 in range(0, cars2):
                                stateTransitonProb = env.poisson(env.r1, r1)*env.poisson(env.r2, r2)*env.poisson(env.m1, m1)*env.poisson(env.m2, m2)
                                c1 = min(max(cars1-action, 0), 21) # This is the cars after the action
                                c2 = min(max(cars2+action, 0), 21 )
                                nextState = (min(max(c1-r1+m1, 0), 21), min(max(c1-r1+m1, 0), 21)) # This is the next state. 
                                stateTransitonProb = env.poisson(env.r1, r1)*env.poisson(env.r2, r2)*env.poisson(env.m1, m1)*env.poisson(env.m2, m2)
                                actionVal += (stateTransitonProb)* ((c1+c2)*env.rentRevenue + gamma*value[nextState])
                actionVal += action*env.moveCost
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

def graphHelper(value: dict, figName: str):
    board = np.zeros((22, 22), dtype=float)

    for car1 in range(22):
        for car2 in range(22):
            board[car1, car2] = round(value[(car1, car2)])

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(board, origin="lower", interpolation="none", aspect="equal")

    # Tick marks at each cell
    ax.set_xticks(range(22))
    ax.set_yticks(range(22))
    ax.set_xlabel("cars1 (x)")
    ax.set_ylabel("cars2 (y)")

    # Cell values
    for y in range(22):
        for x in range(22):
            ax.text(x, y, f"{board[y, x]:g}", ha="center", va="center", fontsize=7)

    # Grid lines on cell boundaries
    ax.set_xticks(np.arange(-.5, 22, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 22, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, label="value")
    plt.tight_layout()
    plt.savefig(figName)
    plt.show()

    time.sleep(1)

def main():
    theta = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    var = init()
    policy_stable = False
    iter = 0
    while not policy_stable:
        var["value"] = policyEval(gamma, theta, **var) # updating the state-values
        var["policy"], policy_stable = policyImprov(gamma, **var) # updating the policy
        graphHelper(var["value"], f'Values: {iter}')
        graphHelper(var["policy"], f'Policy: {iter}')
    print(var["policy"])


if __name__ == "__main__":
    main()
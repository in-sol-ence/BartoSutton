from jackCarRental import JackCarRental
import time as time
import numpy as np
import matplotlib.pyplot as plt

def init() -> dict:
    env = JackCarRental(10, 10, 1)
    print(env.get_state())
    ## Initialize policy and value function
    # Considering it is a deterministic environment, we can use a dictionary to store the policy and value function
    policy = {}
    value = {}
    for cars1 in range(22):
        for cars2 in range(22):
            policy[(cars1, cars2)] = 0  # Initial policy: no cars moved
            value[(cars1, cars2)] = 0.0

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
                v = value[(cars1, cars2)]
                action = policy[(cars1, cars2)]
                expectedRew = min(env.poisson_lambda_rent1, cars1) + min(env.poisson_lambda_rent2, cars2)
                nextState = (max(0, cars1-action-env.poisson_lambda_rent1), max(0, cars2+action-env.poisson_lambda_rent2))
                value[(cars1, cars2)] = expectedRew + gamma*value[nextState]
                delta = max(delta, abs(v-value[(cars1, cars2)]))
        policy_eval += 1
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
            argMax = value[(cars1, cars2)]
            for action in range(-5, 6):
                expectedRew = min(env.poisson_lambda_rent1, cars1) + min(env.poisson_lambda_rent2, cars2)
                nextState = (min(max(0, cars1-action-env.poisson_lambda_rent1), 21), min(max(0, cars2+action-env.poisson_lambda_rent2), 21))
                arg = expectedRew + gamma*value[nextState]
                if arg > argMax:
                    argMax = arg
                    best_action = action
                    policy_stable=False
            policy[(cars1, cars1)] = best_action
                
    if policy_stable:
        print("Found optimal policy")
        return policy, policy_stable 
    else:
        print("Going back in")
        return policy, policy_stable

def graphHelper(value: dict):
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
    plt.show()
    time.sleep(1)

def main():
    theta = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    var = init()
    policy_stable = False
    while not policy_stable:
        var["value"] = policyEval(gamma, theta, **var) # updating the state-values
        var["policy"], policy_stable = policyImprov(gamma, **var) # updating the policy
        graphHelper(var["value"])
    print(var["policy"])


if __name__ == "__main__":
    main()
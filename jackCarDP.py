from jackCarRental import JackCarRental

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
    delta = 0.0
    policy_eval = 0
    while (delta < theta):
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

def policyImprov(gamma, value, policy, env) -> None:
    """Given accurate (to theta) state-value pairs and certain environment dynamics,
    policyImprov finds the best policy pi."""
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
                
    if policy_stable:
        print("Found optimal policy")
    else:
        print("Going back in")

def main():
    theta = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    var = init()
    var["value"] = policyEval(gamma, theta, **var) # updating the state-values
    var["policy"] = policyEval(gamma, **var) # updating the policy




    

if __name__ == "__main__":
    main()
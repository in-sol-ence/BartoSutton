from jackCarRental import JackCarRental

env = JackCarRental(10, 10, 1)

print(env.get_state())



## Initialize policy and value function
# Considering it is a deterministic environment, we can use a dictionary to store the policy and value function

policy = {}
value = {}

for cars1 in range(21):
    for cars2 in range(21):
        policy[(cars1, cars2)] = 0  # Initial policy: no cars moved
        value[(cars1, cars2)] = 0.0

theta = 0.1  # Learning rate
delta = 0.0
gamma = 0.9  # Discount factor
policy_eval = 0
oldpolicy = policy.copy() # This will be used in policy improvement later

# Policy Evaluation
while (delta < theta):
    delta = 0.0
    for cars1 in range(21):
        for cars2 in range(21):
            v = value[(cars1, cars2)]
            action = policy[(cars1, cars2)]
            expectedRew = min(env.poisson_lambda_rent1, cars1) + min(env.poisson_lambda_rent2, cars2)
            nextState = (max(0, cars1-action-env.poisson_lambda_rent1), max(0, cars2+action-env.poisson_lambda_rent2))
            value[(cars1, cars2)] = expectedRew + gamma*value[nextState]
            delta = max(delta, abs(v-value[(cars1, cars2)]))
    policy_eval += 1
    print(f"Policy evaluation {policy_eval} completed with delta: {delta}.")

# Policy Improvement
policy_stable = True
for cars1 in range(21):
    for cars2 in range(21):
        best_action = policy[(cars1, cars2)]
        argMax = value[(cars1, cars2)]
        for action in range(-5, 6):
            expectedRew = min(env.poisson_lambda_rent1, cars1) + min(env.poisson_lambda_rent2, cars2)
            nextState = (max(0, cars1-action-env.poisson_lambda_rent1), max(0, cars2+action-env.poisson_lambda_rent2))
            arg = expectedRew + gamma*value[nextState]
            if arg > argMax:
                argMax = arg
                best_action = action
                policy_stable=False

if policy_stable:
    print("Found optimal policy")
else:
    print("Going back in")     
from jackCarRental import JackCarRental

Rental = JackCarRental(10, 10, 1)

print(Rental.get_state())



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

def simulate_action(state, action):
    ''' Simulate the action and return the new state and reward. '''
    cars1, cars2 = state
    try:
        Rental.location1_cars = cars1
        Rental.location2_cars = cars2
        cost = Rental.action(action)
        reward = Rental.day() - cost
        new_state = (Rental.location1_cars, Rental.location2_cars)
        return new_state, reward
    except ValueError:
        pass

while (delta < theta):
    
    delta = 0.0
    for cars1 in range(21):
        for cars2 in range(21):
            v = value[(cars1, cars2)]
            action = policy[(cars1, cars2)]
            value[(cars1, cars2)] = 0.0
            # Simulate the action
            try:
                state, rew = simulate_action((cars1, cars2), action)
                value[(cars1, cars2)] = rew + gamma * value[state]
            except ValueError:
                print(f"Invalid action {action} for state {(cars1, cars2)}")
                pass
            value[(cars1, cars2)] = rew + gamma * value[state]
            delta = max(delta, abs(v- value[(cars1, cars2)]))
    policy_eval += 1
    print(f"Policy evaluation {policy_eval} completed with delta {delta}.")

# Policy Improvement
old_policy = policy.copy()
policy_stable = True
for cars1 in range(21):
    for cars2 in range(21):
        best_action = policy[(cars1, cars2)]
        for action in range(-5, 6):
            if action > 0 and action > cars1 or action < 0 and -action > cars2:
                continue
            state, rew = simulate_action((cars1, cars2), action)
            

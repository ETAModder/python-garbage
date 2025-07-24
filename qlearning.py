import random

# Define the environment
num_states = 5
num_actions = 3
q_table = [[0 for _ in range(num_actions)] for _ in range(num_states)]
learning_rate = 0.9
discount_factor = 0.99
epsilon = 0.1  # Exploration rate

# Scaling factor for rewards
scaling_factor = 2  # Adjust this factor to scale rewards moderately

# Custom reward function with normalization
def custom_reward_function(state, action):
    if state == 0 and action == 1:
        return 10  # High reward for a specific state-action pair
    return random.uniform(-1, 1) * scaling_factor  # Scaled default reward for other pairs

# Normalizing rewards to ensure they are within a reasonable range
def normalize_reward(reward):
    return max(min(reward, 10), -10)  # Clamping reward between -10 and 10

# Use the custom reward function to set rewards
rewards = [[normalize_reward(custom_reward_function(state, action)) for action in range(num_actions)] for state in range(num_states)]

# Function to choose an action based on the epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return q_table[state].index(max(q_table[state]))

# Function to update Q-values with scaled/custom rewards
def update_q_table(state, action, reward, next_state):
    best_next_action = q_table[next_state].index(max(q_table[next_state]))
    td_target = reward + discount_factor * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += learning_rate * td_error

# Sample training loop
for episode in range(50000):
    state = random.randint(0, num_states - 1)  # Initialize the state
    done = False
    while not done:
        action = choose_action(state)
        next_state = (state + action) % num_states  # Simplified state transition
        reward = rewards[state][action]  # Use predefined reward from the custom reward function
        update_q_table(state, action, reward, next_state)
        state = next_state
        if state == num_states - 1:  # Simplified termination condition
            done = True
    
    # Print Q-table and evaluate performance every 1000 episodes
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}")
        for state in range(num_states):
            action = choose_action(state)
            print(f"State: {state}, Best Action: {action}, Q-Values: {q_table[state]}")
        print()

# Print final Q-Table
print("Final Trained Q-Table:")
for state in q_table:
    print(state)
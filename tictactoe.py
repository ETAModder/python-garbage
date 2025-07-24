import random
import pickle

def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 10)

def check_win(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(row[col] == player for row in board):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def check_draw(board):
    return all(cell != ' ' for row in board for cell in row)

learning_rate = 0.99
discount_factor = 0.9
epsilon = 0.2

# Load Q-tables if they exist, otherwise initialize
try:
    with open('q_table_1.pkl', 'rb') as f:
        q_table_1 = pickle.load(f)
except FileNotFoundError:
    q_table_1 = {}

try:
    with open('q_table_2.pkl', 'rb') as f:
        q_table_2 = pickle.load(f)
except FileNotFoundError:
    q_table_2 = {}

def get_state(board):
    return tuple(tuple(row) for row in board)

def get_q_value(q_table, state, action):
    return q_table.get((state, action), 0)

def set_q_value(q_table, state, action, value):
    q_table[(state, action)] = value

def choose_best_action(board, q_table, player):
    state = get_state(board)
    best_action = None
    best_value = -float('inf')
    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ':
                action = (r, c)
                q_value = get_q_value(q_table, state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
    return best_action

def choose_action(board, q_table, player):
    if random.uniform(0, 1) < epsilon:
        return random.choice([(r, c) for r in range(3) for c in range(3) if board[r][c] == ' '])
    else:
        return choose_best_action(board, q_table, player)

def update_q_table(q_table, state, action, reward, next_state, next_action):
    current_q_value = get_q_value(q_table, state, action)
    next_q_value = get_q_value(q_table, next_state, next_action)
    new_q_value = current_q_value + learning_rate * (reward + discount_factor * next_q_value - current_q_value)
    set_q_value(q_table, state, action, new_q_value)

def reward_ai_1(board, player):
    if check_win(board, player):
        return 10 + random.uniform(-0.5, 0.5)
    elif check_win(board, 'X' if player == 'O' else 'O'):
        return -10 + random.uniform(-0.5, 0.5)
    else:
        return 0 + random.uniform(-0.1, 0.1)

def reward_ai_2(board, player):
    if check_win(board, player):
        return 5 + random.uniform(-0.5, 0.5)
    elif check_win(board, 'X' if player == 'O' else 'O'):
        return -5 + random.uniform(-0.5, 0.5)
    else:
        return 1 + random.uniform(-0.1, 0.1)

def play_training_game():
    board = initialize_board()
    current_player = 'X'
    opponent = 'O'
    
    state_action_rewards = []

    while True:
        if current_player == 'X':
            action = choose_action(board, q_table_1, current_player)
            reward_fn = reward_ai_1
        else:
            action = choose_action(board, q_table_2, current_player)
            reward_fn = reward_ai_2
            
        state = get_state(board)
        board[action[0]][action[1]] = current_player
        reward = reward_fn(board, current_player)
        state_action_rewards.append((state, action, reward))
        
        if check_win(board, current_player) or check_draw(board):
            break
        current_player, opponent = opponent, current_player
    
    return state_action_rewards

# Train both AIs
for _ in range(10000):
    state_action_rewards = play_training_game()
    current_player = 'X'
    opponent = 'O'
    for i in range(len(state_action_rewards) - 1):
        state, action, reward = state_action_rewards[i]
        next_state, next_action, _ = state_action_rewards[i + 1]
        update_q_table(q_table_1 if current_player == 'X' else q_table_2, state, action, reward, next_state, next_action)
        current_player, opponent = opponent, current_player
    final_state, final_action, final_reward = state_action_rewards[-1]
    set_q_value(q_table_1 if current_player == 'X' else q_table_2, final_state, final_action, final_reward)

# Save Q-tables after training
with open('q_table_1.pkl', 'wb') as f:
    pickle.dump(q_table_1, f)

with open('q_table_2.pkl', 'wb') as f:
    pickle.dump(q_table_2, f)

def play_game():
    board = initialize_board()
    print_board(board)
    current_player = 'X'
    opponent = 'O'
    
    for turn in range(9):
        if current_player == 'X':
            move = choose_best_action(board, q_table_1, current_player)
        else:
            move = choose_best_action(board, q_table_2, current_player)
        
        if move:
            board[move[0]][move[1]] = current_player
            print_board(board)
            if check_win(board, current_player):
                print(f"Player {current_player} wins!")
                return
            current_player, opponent = opponent, current_player
    
    print("It's a draw!")

play_game()

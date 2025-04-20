import numpy as np
import time
from gomoku_env import Gomoku

# Constants for the players
PLAYER_1 = 1
PLAYER_2 = -1
WINNING_LENGTH = 4
BOARD_SIZE = 5

# Global variable for maximum time
MAX_TIME = 5.0  # Set the time limit (in seconds)

# Directions for checking rows, columns, and diagonals
DIRECTIONS = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]

def is_winning(board, player, game):
    player_board = np.zeros(BOARD_SIZE, dtype=np.uint16)
    board = np.copy(board)
    board[board != player] = 0
    if player == -1:
        board = -board
    # Process the flattened array in chunks of 16 bits
    for i in range(BOARD_SIZE):
        # Use np.dot to calculate the integer value of the 16-bit sequence in the row
        player_board[i] = np.dot(board[i], 2**np.arange(BOARD_SIZE-1, -1, -1))
    #if game.check_win(player_board):
    #    print(player_board)
    #    print(board)
    #    raise ValueError()
    #if game.check_win(player_board) and player == -1:
    #    print(player_board)
    #    print(board)

    return game.check_win(player_board)

def get_valid_moves(board):
    return [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if board[x][y] == 0]

def count_threats(board, player):
    """Count the number of potential threats for the player to form a line of 4 pieces."""
    threats = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == player:
                for dx, dy in DIRECTIONS:
                    count = 1
                    for i in range(1, WINNING_LENGTH):
                        nx, ny = x + dx * i, y + dy * i
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == player:
                            count += 1
                        else:
                            break
                    if count >= 3:
                        threats += 1
    return threats


def evaluate(board, game, player):
    """Evaluate the board state using a more advanced heuristic."""
    # Priority 1: Check if the current player has won
    if is_winning(board, player, game):
        return 1000
    elif is_winning(board, -player, game):
        return -1000
    # Priority 2: Count the number of potential threats for both players
    player_1_threats = count_threats(board, player)
    player_2_threats = count_threats(board, -player)

    # Calculate the heuristic based on potential threats
    heuristic_score = player_1_threats - player_2_threats
    
    # Priority 3: Control of the center (center is the most strategic area)
    center = [(2, 2)]
    for x, y in center:
        if board[x][y] == player:
            heuristic_score += 10
        elif board[x][y] == -player:
            heuristic_score -= 10

    return heuristic_score


def negamax(board, alpha, beta, player, game, depth, start_time):
    if is_winning(board, player, game):
        return 1000, None
    elif is_winning(board, -player, game):
        return -1000, None

    valid_moves = get_valid_moves(board)
    
    if time.time() - start_time > MAX_TIME:
        return None, None

    # Stop the search if time limit is reached
    if depth == 0 or not valid_moves:
        return evaluate(board, game, player), None

    max_eval = -float('inf')
    best_move = None
    ori_board = board
    for move in valid_moves:
        board = np.copy(ori_board)
        x, y = move
        board[x][y] = player
        eval_score, _ = negamax(board, -beta, -alpha, -player, game, depth-1, start_time)
        if eval_score == None:
            return None, None
        board[x][y] = 0
        eval_score = -eval_score

        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move

        alpha = max(alpha, eval_score)
        if alpha >= beta:
            #if depth == 7:
            #    print(alpha, beta)
            break

    return max_eval, best_move

def iterative_deepening_search(board, player, game):
    """Iterative Deepening Search with Negamax."""
    start_time = time.time()
    best_move = None
    depth = 1
    best_value = -float('inf')

    # Iteratively deepen the search
    while time.time() - start_time < MAX_TIME:
        if depth > BOARD_SIZE * BOARD_SIZE:
            break
        # Run Negamax with the current depth
        value, move = negamax(board, -float('inf'), float('inf'), player, game, depth, start_time)
        if value == None:
            break
        # Track the best move found at the current depth
        if value > best_value:
            best_value = value
            best_move = move
        
        # Increase the depth for the next iteration
        depth += 1
    #print("Max depth", depth-1, best_value)
    return best_move


def gomoku_ai(board, g):
    game = g
    best_move = iterative_deepening_search(board, PLAYER_2, g)
    return best_move

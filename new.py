import numpy as np

ROWS = 6
COLS = 7

# Constants for evaluation
WIN_SCORE = float('inf')
LOSS_SCORE = float('-inf')

# Constants for transposition table
TRANSPOSITION_EXACT = 0
TRANSPOSITION_UPPERBOUND = 1
TRANSPOSITION_LOWERBOUND = 2

# Initialize transposition table
transposition_table = {}


# Function to print the Connect4 board
def print_board(board):
    for row in board:
        print(' '.join(row))


# Function to check if a player has won
def check_win(board, player):
    # Check horizontally
    for i in range(ROWS):
        for j in range(COLS - 3):
            if board[i][j] == player and board[i][j + 1] == player and board[i][j + 2] == player and board[i][j + 3] == player:
                return True

    # Check vertically
    for i in range(ROWS - 3):
        for j in range(COLS):
            if board[i][j] == player and board[i + 1][j] == player and board[i + 2][j] == player and board[i + 3][j] == player:
                return True

    # Check diagonally (positive slope)
    for i in range(ROWS - 3):
        for j in range(COLS - 3):
            if board[i][j] == player and board[i + 1][j + 1] == player and board[i + 2][j + 2] == player and board[i + 3][j + 3] == player:
                return True

    # Check diagonally (negative slope)
    for i in range(3, ROWS):
        for j in range(COLS - 3):
            if board[i][j] == player and board[i - 1][j + 1] == player and board[i - 2][j + 2] == player and board[i - 3][j + 3] == player:
                return True

    return False


# Function to check if the game ended in a draw
def check_draw(board):
    return all(board[i][j] != ' ' for i in range(ROWS) for j in range(COLS))


# Function to evaluate the board for the minimax algorithm
def evaluate_board(board):
    # Evaluate horizontally
    for i in range(ROWS):
        for j in range(COLS - 3):
            if board[i][j] == 'X' and board[i][j + 1] == 'X' and board[i][j + 2] == 'X' and board[i][j + 3] == 'X':
                return WIN_SCORE
            elif board[i][j] == 'O' and board[i][j + 1] == 'O' and board[i][j + 2] == 'O' and board[i][j + 3] == 'O':
                return LOSS_SCORE

    # Evaluate vertically
    for i in range(ROWS - 3):
        for j in range(COLS):
            if board[i][j] == 'X' and board[i + 1][j] == 'X' and board[i + 2][j] == 'X' and board[i + 3][j] == 'X':
                return WIN_SCORE
            elif board[i][j] == 'O' and board[i + 1][j] == 'O' and board[i + 2][j] == 'O' and board[i + 3][j] == 'O':
                return LOSS_SCORE

    # Evaluate diagonally (positive slope)
    for i in range(ROWS - 3):
        for j in range(COLS - 3):
            if board[i][j] == 'X' and board[i + 1][j + 1] == 'X' and board[i + 2][j + 2] == 'X' and board[i + 3][j + 3] == 'X':
                return WIN_SCORE
            elif board[i][j] == 'O' and board[i + 1][j + 1] == 'O' and board[i + 2][j + 2] == 'O' and board[i + 3][j + 3] == 'O':
                return LOSS_SCORE

    # Evaluate diagonally (negative slope)
    for i in range(3, ROWS):
        for j in range(COLS - 3):
            if board[i][j] == 'X' and board[i - 1][j + 1] == 'X' and board[i - 2][j + 2] == 'X' and board[i - 3][j + 3] == 'X':
                return WIN_SCORE
            elif board[i][j] == 'O' and board[i - 1][j + 1] == 'O' and board[i - 2][j + 2] == 'O' and board[i - 3][j + 3] == 'O':
                return LOSS_SCORE

    return 0


# Function for AI's turn using minimax algorithm with alpha-beta pruning and enhancements
def minimax(board, depth, alpha, beta, maximizingPlayer):
    score = evaluate_board(board)

    if score != 0:
        return score

    if depth == 0 or check_draw(board):
        return 0

    # Transposition table lookup
    board_hash = tuple(map(tuple, board))
    if board_hash in transposition_table:
        entry = transposition_table[board_hash]
        if entry['depth'] >= depth:
            if entry['type'] == TRANSPOSITION_EXACT:
                return entry['score']
            elif entry['type'] == TRANSPOSITION_LOWERBOUND:
                alpha = max(alpha, entry['score'])
            elif entry['type'] == TRANSPOSITION_UPPERBOUND:
                beta = min(beta, entry['score'])
            if alpha >= beta:
                return entry['score']

    if maximizingPlayer:
        max_score = LOSS_SCORE

        # Generate all possible moves and order them
        possible_moves = []
        for col in range(COLS):
            if board[0][col] == ' ':
                for row in range(ROWS - 1, -1, -1):
                    if board[row][col] == ' ':
                        possible_moves.append((row, col))
                        break

        possible_moves.sort(key=lambda move: move[0], reverse=True)  # Order moves by row (descending)

        for move in possible_moves:
            row, col = move
            board[row][col] = 'X'
            current_score = minimax(board, depth - 1, alpha, beta, False)
            board[row][col] = ' '
            max_score = max(max_score, current_score)
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break

        # Store entry in transposition table
        transposition_table[board_hash] = {
            'type': TRANSPOSITION_LOWERBOUND,
            'depth': depth,
            'score': max_score
        }

        return max_score
    else:
        min_score = WIN_SCORE

        # Generate all possible moves and order them
        possible_moves = []
        for col in range(COLS):
            if board[0][col] == ' ':
                for row in range(ROWS - 1, -1, -1):
                    if board[row][col] == ' ':
                        possible_moves.append((row, col))
                        break

        possible_moves.sort(key=lambda move: move[0], reverse=True)  # Order moves by row (descending)

        for move in possible_moves:
            row, col = move
            board[row][col] = 'O'
            current_score = minimax(board, depth - 1, alpha, beta, True)
            board[row][col] = ' '
            min_score = min(min_score, current_score)
            beta = min(beta, min_score)
            if alpha >= beta:
                break

        # Store entry in transposition table
        transposition_table[board_hash] = {
            'type': TRANSPOSITION_UPPERBOUND,
            'depth': depth,
            'score': min_score
        }

        return min_score


# Function to make a move for the AI
def make_move(board):
    max_score = LOSS_SCORE
    best_move = None

    # Generate all possible moves and order them
    possible_moves = []
    for col in range(COLS):
        if board[0][col] == ' ':
            for row in range(ROWS - 1, -1, -1):
                if board[row][col] == ' ':
                    possible_moves.append((row, col))
                    break

    possible_moves.sort(key=lambda move: move[0], reverse=True)  # Order moves by row (descending)

    for move in possible_moves:
        row, col = move
        board[row][col] = 'X'
        score = minimax(board, 6, LOSS_SCORE, WIN_SCORE, False)
        board[row][col] = ' '
        if score > max_score:
            max_score = score
            best_move = move

    row, col = best_move
    board[row][col] = 'X'


# Function to play the Connect4 game
def play_game():
    board = np.full((ROWS, COLS), ' ')
    print("Welcome to Connect4!")

    while True:
        print_board(board)

        # Player's turn
        col = int(input("Enter the column number (1-7) to make a move: ")) - 1
        if col < 0 or col >= COLS:
            print("Invalid column number. Please try again.")
            continue

        if board[0][col] != ' ':
            print("Column is already full. Please try again.")
            continue

        for row in range(ROWS - 1, -1, -1):
            if board[row][col] == ' ':
                board[row][col] = 'O'
                break

        if check_win(board, 'O'):
            print_board(board)
            print("Congratulations! You win!")
            break

        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        # AI's turn
        make_move(board)

        if check_win(board, 'X'):
            print_board(board)
            print("AI wins!")
            break

        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break


# Start the game
play_game()

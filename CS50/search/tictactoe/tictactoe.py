"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board: list[list]):
    """
    Returns player who has the next turn on a board.
    """
    count = sum(1 if cell == EMPTY else 0 for row in board for cell in row)
    return X if count % 2 == 1 else O


def actions(board: list[list]):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = [(i, j) for i, row in enumerate(board)
               for j, cell in enumerate(row) if cell == EMPTY]
    return actions


def result(board: list[list], action: tuple[int, int]):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] != EMPTY:
        raise Exception("Invalid action")
    result = [row[:] for row in board]
    result[i][j] = player(result)
    return result


def winner(board: list[list]):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(len(board)):
        same_in_row = board[i][0] == board[i][1] and board[i][1] == board[i][2]
        if same_in_row and board[i][0] != EMPTY:
            return board[i][0]

        same_in_col = board[0][i] == board[1][i] and board[1][i] == board[2][i]
        if same_in_col and board[0][i] != EMPTY:
            return board[0][i]

    same_in_diag = board[0][0] == board[1][1] and board[1][1] == board[2][2] # \
    if same_in_diag and board[0][0] != EMPTY:
        return board[0][0]

    same_in_diag = board[0][2] == board[1][1] and board[1][1] == board[2][0] # \
    if same_in_diag and board[0][2] != EMPTY:
        return board[0][2]

    return None


def terminal(board: list[list]):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    count = sum(1 if cell == EMPTY else 0 for row in board for cell in row)
    return count == 0


def utility(board: list[list]):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    match winner(board):
        case "X": return 1
        case "O": return -1
        case _: return 0


def score(board: list[list]):
    if terminal(board):
        return utility(board)

    p = player(board)
    fn = max if p == X else min
    best = -2 if p == X else +2
    for action in actions(board):
        new_board = result(board, action)
        best = fn(best, score(new_board))
    return best


def alphabeta(board: list[list], alpha: int, beta: int):
    if terminal(board):
        return utility(board)

    p = player(board)
    best = -2 if p == X else +2
    for action in actions(board):
        s = alphabeta(result(board, action), beta, alpha)
        if p == X:
            best = max(best, s)
            alpha = max(alpha, s)
            if beta <= alpha:
                return alpha
        else:
            best = min(best, s)
            alpha = min(alpha, s)
            if alpha <= beta:
                return alpha
    return best



def minimax(board: list[list]):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    # if player(board) == X:
        # best_action = max(actions(board), key=lambda a: score(result(board, a)))
    # else:
        # best_action = min(actions(board), key=lambda a: score(result(board, a)))
    # return best_action

    if player(board) == X:
        best_action = max(actions(board), key=lambda a: alphabeta(result(board, a), +1, -1))
    else:
        best_action = min(actions(board), key=lambda a: alphabeta(result(board, a), -1, +1))
    return best_action

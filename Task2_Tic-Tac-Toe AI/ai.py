import math

class TicTacToeAI:
    """Implements the Minimax algorithm with Alpha-Beta Pruning to make unbeatable decisions."""

    def __init__(self, ai_symbol, human_symbol):
        """
        Initializes the AI agent.
        Args:
            ai_symbol (str): Symbol the AI plays ('X' or 'O').
            human_symbol (str): Symbol the human plays ('X' or 'O').
        """
        self.ai_symbol = ai_symbol
        self.human_symbol = human_symbol

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """
        Recursively evaluates all possible board states to find the optimal score.
        Uses Alpha-Beta Pruning to prune branches that cannot influence the final decision.

        Args:
            board (Board): The current board state.
            depth (int): The current depth in the search tree.
            alpha (float): Best already explored option along path to root for maximizer.
            beta (float): Best already explored option along path to root for minimizer.
            is_maximizing (bool): True if maximizing score (AI turn), False if minimizing (Human turn).

        Returns:
            int: The evaluated score of the board state.
        """
        # 1. Terminal State Evaluations
        winner = board.check_winner()
        if winner == self.ai_symbol:
            # Positive score for AI win, subtract depth to favor quicker wins
            return 10 - depth
        elif winner == self.human_symbol:
            # Negative score for human win, add depth to favor longer survival
            return -10 + depth
        elif board.is_draw():
            # Zero score for a draw
            return 0

        # 2. Recursive Minimax step
        if is_maximizing:
            max_eval = -math.inf
            # Iterate through all available moves
            for row, col in board.get_empty_cells():
                # Simulate move
                board.make_move(row, col, self.ai_symbol)
                # Recurse
                score = self.minimax(board, depth + 1, alpha, beta, False)
                # Undo simulated move
                board.undo_move(row, col)
                
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                
                # Alpha-Beta pruning condition
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            # Iterate through all available moves
            for row, col in board.get_empty_cells():
                # Simulate move
                board.make_move(row, col, self.human_symbol)
                # Recurse
                score = self.minimax(board, depth + 1, alpha, beta, True)
                # Undo simulated move
                board.undo_move(row, col)
                
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                
                # Alpha-Beta pruning condition
                if beta <= alpha:
                    break
            return min_eval

    def best_move(self, board):
        """
        Evaluates the optimal move for the AI using the Minimax algorithm.

        Args:
            board (Board): The current board state.

        Returns:
            tuple: Coordinates of the best move (row, col) or None if no moves available.
        """
        best_score = -math.inf
        move = None
        empty_cells = board.get_empty_cells()

        # Iterate over all empty cells to find the one with the highest score
        for row, col in empty_cells:
            # Simulate placing AI's symbol
            board.make_move(row, col, self.ai_symbol)
            # Evaluate using minimax starting at depth 0
            score = self.minimax(board, 0, -math.inf, math.inf, False)
            # Undo simulated move
            board.undo_move(row, col)

            if score > best_score:
                best_score = score
                move = (row, col)

        return move

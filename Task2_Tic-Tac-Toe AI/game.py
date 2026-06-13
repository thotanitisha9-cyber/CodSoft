import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform ANSI support
colorama.init(autoreset=True)


class Board:
    """Represents the 3x3 Tic-Tac-Toe board."""

    def __init__(self):
        # Initialize an empty 3x3 board grid
        self.grid = [[" " for _ in range(3)] for _ in range(3)]

    def get_cell(self, row, col):
        """Returns the value at the given row and column (0-indexed)."""
        return self.grid[row][col]

    def set_cell(self, row, col, symbol):
        """Sets the cell value at the given row and column (0-indexed)."""
        self.grid[row][col] = symbol

    def is_valid_move(self, row, col):
        """Checks if a move is within bounds and the cell is empty."""
        return 0 <= row < 3 and 0 <= col < 3 and self.grid[row][col] == " "

    def make_move(self, row, col, symbol):
        """Places a symbol on the board. Returns True if successful, False otherwise."""
        if self.is_valid_move(row, col):
            self.set_cell(row, col, symbol)
            return True
        return False

    def undo_move(self, row, col):
        """Removes a symbol from the board (used for Minimax simulation)."""
        if 0 <= row < 3 and 0 <= col < 3:
            self.set_cell(row, col, " ")

    def get_empty_cells(self):
        """Returns a list of coordinate tuples (row, col) that are empty."""
        empty = []
        for r in range(3):
            for c in range(3):
                if self.grid[r][c] == " ":
                    empty.append((r, c))
        return empty

    def is_full(self):
        """Checks if the board is completely filled."""
        return len(self.get_empty_cells()) == 0

    def check_winner(self):
        """
        Checks the board for a winner.
        Returns:
            'X' or 'O' if there is a winner,
            None if no winner yet.
        """
        # Check rows
        for r in range(3):
            if self.grid[r][0] == self.grid[r][1] == self.grid[r][2] != " ":
                return self.grid[r][0]

        # Check columns
        for c in range(3):
            if self.grid[0][c] == self.grid[1][c] == self.grid[2][c] != " ":
                return self.grid[0][c]

        # Check diagonals
        if self.grid[0][0] == self.grid[1][1] == self.grid[2][2] != " ":
            return self.grid[0][0]
        if self.grid[0][2] == self.grid[1][1] == self.grid[2][0] != " ":
            return self.grid[0][2]

        return None

    def is_draw(self):
        """Checks if the game is a draw (board is full and there is no winner)."""
        return self.is_full() and self.check_winner() is None

    def display(self):
        """
        Renders the board to the console with rich aesthetics.
        Uses box-drawing characters and color formatting.
        """
        # Stylized colored symbols
        def fmt_cell(val, r, c):
            if val == "X":
                return f"{Fore.CYAN}{Style.BRIGHT}X{Style.RESET_ALL}"
            elif val == "O":
                return f"{Fore.YELLOW}{Style.BRIGHT}O{Style.RESET_ALL}"
            else:
                # Subtly show the cell coordinates for user convenience if empty
                return f"{Fore.WHITE}{Style.DIM}{r+1},{c+1}{Style.RESET_ALL}"

        print(f"\n    {Fore.GREEN}1      2      3{Style.RESET_ALL}")
        print(f"  ┌──────┬──────┬──────┐")
        for r in range(3):
            row_str = f"{Fore.GREEN}{r+1}{Style.RESET_ALL} │"
            for c in range(3):
                val = self.grid[r][c]
                row_str += f"  {fmt_cell(val, r, c)}  │"
            print(row_str)
            if r < 2:
                print(f"  ├──────┼──────┼──────┤")
        print(f"  └──────┴──────┴──────┘\n")


class TicTacToe:
    """Manages the overall game logic and player turns."""

    def __init__(self, human_symbol):
        self.board = Board()
        self.human_symbol = human_symbol.upper()
        self.ai_symbol = "O" if self.human_symbol == "X" else "X"
        self.current_turn = "X"  # 'X' always starts in standard Tic-Tac-Toe

    def switch_turn(self):
        """Switches the current active player turn."""
        self.current_turn = "O" if self.current_turn == "X" else "X"

    def get_current_player_symbol(self):
        """Returns the symbol of the player whose turn it is."""
        return self.current_turn

    def is_game_over(self):
        """Returns True if the game has ended in a win or draw."""
        return self.board.check_winner() is not None or self.board.is_draw()

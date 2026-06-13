import sys
import time
from game import Board, TicTacToe
from ai import TicTacToeAI
from colorama import Fore, Style, init

# Reconfigure stdout to use UTF-8 if running on Windows to prevent UnicodeEncodeError with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Initialize colorama
init(autoreset=True)

def print_header():
    """Prints a beautiful title header for the terminal game."""
    header = f"""
{Fore.CYAN}{Style.BRIGHT}==============================================
         TIC-TAC-TOE UNBEATABLE AI
=============================================={Style.RESET_ALL}
Welcome to Tic-Tac-Toe! 
You will face an AI opponent using the Minimax search
algorithm optimized with Alpha-Beta Pruning.
Can you force a draw? The AI is mathematically unbeatable!
"""
    print(header)

def get_human_symbol():
    """Prompts the user to choose their symbol (X or O) with input validation."""
    while True:
        choice = input(f"Choose your symbol ({Fore.CYAN}X{Style.RESET_ALL}/{Fore.YELLOW}O{Style.RESET_ALL}): ").strip().upper()
        if choice in ["X", "O"]:
            return choice
        print(f"{Fore.RED}Invalid input. Please enter 'X' or 'O'.{Style.RESET_ALL}")

def get_human_move(board):
    """
    Prompts the user for their move coordinates, validates the input,
    and returns a tuple of 0-indexed (row, col).
    """
    while True:
        try:
            move_input = input(f"Enter row and column ({Fore.GREEN}1-3 1-3{Style.RESET_ALL}) separated by a space (or 'q' to quit): ").strip()
            
            if move_input.lower() == 'q':
                print(f"\n{Fore.RED}Exiting the game. Thanks for playing!{Style.RESET_ALL}")
                sys.exit(0)

            parts = move_input.split()
            if len(parts) != 2:
                raise ValueError("Must provide exactly two numbers.")

            row = int(parts[0]) - 1
            col = int(parts[1]) - 1

            if not (0 <= row < 3 and 0 <= col < 3):
                print(f"{Fore.RED}Coordinates out of range. Row and Column must be between 1 and 3.{Style.RESET_ALL}")
                continue

            if not board.is_valid_move(row, col):
                print(f"{Fore.RED}That cell is already occupied! Choose an empty cell (labeled with coordinates).{Style.RESET_ALL}")
                continue

            return row, col
        except ValueError:
            print(f"{Fore.RED}Invalid input format. Please enter two numbers separated by a space (e.g., '1 2' for row 1, col 2).{Style.RESET_ALL}")

def play_game():
    """Runs a single full game of Tic-Tac-Toe against the AI."""
    print_header()
    human_symbol = get_human_symbol()
    
    # Initialize game state and AI
    game = TicTacToe(human_symbol)
    ai = TicTacToeAI(game.ai_symbol, game.human_symbol)

    print(f"\nGame starts! You are {Fore.CYAN if human_symbol == 'X' else Fore.YELLOW}{human_symbol}{Style.RESET_ALL}.")
    print(f"AI is {Fore.CYAN if game.ai_symbol == 'X' else Fore.YELLOW}{game.ai_symbol}{Style.RESET_ALL}.")
    print(f"'{Fore.CYAN}X{Style.RESET_ALL}' goes first.")

    # Show initial empty board
    game.board.display()

    # Main game loop
    while not game.is_game_over():
        current_player = game.get_current_player_symbol()

        if current_player == game.human_symbol:
            # Human's turn
            print(f"{Fore.GREEN}{Style.BRIGHT}👉 Your Turn ({game.human_symbol}):{Style.RESET_ALL}")
            row, col = get_human_move(game.board)
            game.board.make_move(row, col, game.human_symbol)
        else:
            # AI's turn
            print(f"{Fore.RED}{Style.BRIGHT}🤖 AI is thinking ({game.ai_symbol})...{Style.RESET_ALL}")
            # Add a small aesthetic delay to make the AI feel natural
            time.sleep(0.6)
            row, col = ai.best_move(game.board)
            if row is not None and col is not None:
                game.board.make_move(row, col, game.ai_symbol)
                print(f"AI plays at Row {row+1}, Col {col+1}")
            else:
                print(f"{Fore.RED}AI could not determine a move.{Style.RESET_ALL}")

        # Display the board after the move
        game.board.display()
        
        # Check game status before switching turns
        winner = game.board.check_winner()
        if winner:
            if winner == game.human_symbol:
                print(f"{Fore.GREEN}{Style.BRIGHT}🏆 Unbelievable! You won!{Style.RESET_ALL} (Minimax implementation might have an issue!)")
            else:
                print(f"{Fore.RED}{Style.BRIGHT}💀 Game Over! AI Wins!{Style.RESET_ALL} Better luck next time.")
            break
        elif game.board.is_draw():
            print(f"{Fore.YELLOW}{Style.BRIGHT}🤝 It's a draw! Well played!{Style.RESET_ALL}")
            break

        # Switch turns
        game.switch_turn()

def main():
    """Main function handling replays."""
    try:
        while True:
            play_game()
            
            # Replay prompt
            while True:
                replay = input(f"Do you want to play again? ({Fore.GREEN}y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}): ").strip().lower()
                if replay in ['y', 'yes']:
                    print("\n" + "=" * 46 + "\n")
                    break
                elif replay in ['n', 'no']:
                    print(f"\n{Fore.CYAN}Thanks for playing! Goodbye!{Style.RESET_ALL}")
                    return
                else:
                    print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}Game interrupted. Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

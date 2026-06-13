import tkinter as tk
from tkinter import messagebox
import time
from game import Board
from ai import TicTacToeAI

# Design Styling Constants (Dark Mode Premium Aesthetic)
BG_COLOR = "#1A1B20"         # Dark slate/charcoal background
CARD_BG = "#24252D"          # Slightly lighter panel background
BUTTON_BG = "#2E303D"        # Cell background
BUTTON_HOVER = "#3B3E4F"     # Cell hover color
TEXT_LIGHT = "#E2E8F0"       # Light gray text
COLOR_X = "#00F5FF"          # Vibrant Cyan for X
COLOR_O = "#FFD700"          # Vibrant Gold for O
FONT_FAMILY = "Helvetica"


class TicTacToeGUI:
    """Graphical User Interface for the Tic-Tac-Toe AI game using Tkinter."""

    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe AI (Unbeatable)")
        self.root.geometry("420x540")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        # Center the window on screen
        self.center_window()

        self.human_symbol = None
        self.ai_symbol = None
        self.board = None
        self.ai = None
        self.current_turn = None
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.main_frame = None

        # Show symbol selection screen first
        self.show_symbol_selection()

    def center_window(self):
        """Centers the window on the desktop screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def clear_window(self):
        """Helper to clear all widgets currently inside the window."""
        if self.main_frame:
            self.main_frame.destroy()
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_symbol_selection(self):
        """Displays the splash screen to select X or O."""
        self.clear_window()

        # Selection Frame container
        self.main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(expand=True, fill="both", padx=30, pady=30)

        # Game Title Header
        title_label = tk.Label(
            self.main_frame, 
            text="TIC-TAC-TOE AI", 
            font=(FONT_FAMILY, 28, "bold"), 
            bg=BG_COLOR, 
            fg=COLOR_X
        )
        title_label.pack(pady=(20, 10))

        subtitle_label = tk.Label(
            self.main_frame, 
            text="Unbeatable Minimax Agent", 
            font=(FONT_FAMILY, 14, "italic"), 
            bg=BG_COLOR, 
            fg=TEXT_LIGHT
        )
        subtitle_label.pack(pady=(0, 40))

        prompt_label = tk.Label(
            self.main_frame, 
            text="Choose your symbol:", 
            font=(FONT_FAMILY, 12, "bold"), 
            bg=BG_COLOR, 
            fg=TEXT_LIGHT
        )
        prompt_label.pack(pady=10)

        # Frame for selection buttons
        btn_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        btn_frame.pack(pady=20)

        # Play as X Button
        btn_x = tk.Button(
            btn_frame, 
            text="Play as X\n(Goes First)", 
            font=(FONT_FAMILY, 12, "bold"),
            bg=BUTTON_BG, 
            fg=COLOR_X, 
            activebackground=BUTTON_HOVER, 
            activeforeground=COLOR_X,
            width=12, 
            height=3, 
            bd=0, 
            cursor="hand2",
            relief="flat",
            command=lambda: self.start_game("X")
        )
        btn_x.pack(side="left", padx=15)
        # Add subtle hover colors
        btn_x.bind("<Enter>", lambda e: btn_x.config(bg=BUTTON_HOVER))
        btn_x.bind("<Leave>", lambda e: btn_x.config(bg=BUTTON_BG))

        # Play as O Button
        btn_o = tk.Button(
            btn_frame, 
            text="Play as O\n(Goes Second)", 
            font=(FONT_FAMILY, 12, "bold"),
            bg=BUTTON_BG, 
            fg=COLOR_O, 
            activebackground=BUTTON_HOVER, 
            activeforeground=COLOR_O,
            width=12, 
            height=3, 
            bd=0, 
            cursor="hand2",
            relief="flat",
            command=lambda: self.start_game("O")
        )
        btn_o.pack(side="right", padx=15)
        btn_o.bind("<Enter>", lambda e: btn_o.config(bg=BUTTON_HOVER))
        btn_o.bind("<Leave>", lambda e: btn_o.config(bg=BUTTON_BG))

        # Info footer
        footer_label = tk.Label(
            self.main_frame, 
            text="Note: AI uses Minimax with Alpha-Beta Pruning.", 
            font=(FONT_FAMILY, 9), 
            bg=BG_COLOR, 
            fg="#6B7280"
        )
        footer_label.pack(side="bottom", pady=10)

    def start_game(self, human_symbol):
        """Initializes the board and AI classes and loads the game board GUI."""
        self.human_symbol = human_symbol
        self.ai_symbol = "O" if human_symbol == "X" else "X"
        self.board = Board()
        self.ai = TicTacToeAI(self.ai_symbol, self.human_symbol)
        
        # 'X' always starts
        self.current_turn = "X"

        self.clear_window()

        # Setup game screen frame
        self.main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Status indicators (Header)
        self.status_label = tk.Label(
            self.main_frame, 
            text="Game Started", 
            font=(FONT_FAMILY, 14, "bold"), 
            bg=BG_COLOR, 
            fg=TEXT_LIGHT
        )
        self.status_label.pack(pady=10)

        # 3x3 Grid Board Container
        board_container = tk.Frame(self.main_frame, bg="#111217", padx=3, pady=3)
        board_container.pack(pady=10)

        # Generate board cells
        for r in range(3):
            for c in range(3):
                btn = tk.Button(
                    board_container,
                    text="",
                    font=(FONT_FAMILY, 24, "bold"),
                    bg=BUTTON_BG,
                    fg=TEXT_LIGHT,
                    activebackground=BUTTON_HOVER,
                    activeforeground=TEXT_LIGHT,
                    width=5,
                    height=2,
                    bd=0,
                    relief="flat",
                    command=lambda row=r, col=c: self.on_cell_click(row, col)
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                
                # Bind hover events for nice dynamic feedback
                btn.bind("<Enter>", lambda e, b=btn: self.on_btn_hover(b))
                btn.bind("<Leave>", lambda e, b=btn: self.on_btn_leave(b))
                
                self.buttons[r][c] = btn

        # Turn indicator info
        self.turn_label = tk.Label(
            self.main_frame,
            text=f"You: {self.human_symbol}  |  AI: {self.ai_symbol}",
            font=(FONT_FAMILY, 11, "bold"),
            bg=BG_COLOR,
            fg="#94A3B8"
        )
        self.turn_label.pack(pady=10)

        # If AI plays first (AI is X)
        if self.ai_symbol == "X":
            self.trigger_ai_turn()
        else:
            self.status_label.config(text="👉 Your Turn (X)")

    def on_btn_hover(self, btn):
        """Sets cell button background to hover state if empty."""
        if btn["text"] == "":
            btn.config(bg=BUTTON_HOVER)

    def on_btn_leave(self, btn):
        """Restores cell button background when mouse leaves."""
        if btn["text"] == "":
            btn.config(bg=BUTTON_BG)

    def on_cell_click(self, row, col):
        """Handles human cell clicks."""
        # Only accept click if it's the human's turn and the cell is empty
        if self.current_turn != self.human_symbol:
            return
        if self.board.get_cell(row, col) != " ":
            return

        # Place Human symbol on backend board
        self.board.make_move(row, col, self.human_symbol)
        
        # Update button text and styling
        btn = self.buttons[row][col]
        color = COLOR_X if self.human_symbol == "X" else COLOR_O
        btn.config(text=self.human_symbol, fg=color, bg=CARD_BG, state="disabled")

        if self.check_game_over():
            return

        # Switch turn to AI
        self.current_turn = self.ai_symbol
        self.trigger_ai_turn()

    def trigger_ai_turn(self):
        """Disables controls, updates status, and schedules AI turn with delay."""
        self.status_label.config(text=f"🤖 AI is thinking ({self.ai_symbol})...")
        
        # Disable all empty buttons so user cannot click during AI thinking phase
        self.set_board_interactive_state(False)

        # Schedule AI move with natural aesthetic delay (400ms)
        self.root.after(450, self.execute_ai_move)

    def execute_ai_move(self):
        """Calculates and executes AI's best move."""
        row, col = self.ai.best_move(self.board)
        if row is not None and col is not None:
            self.board.make_move(row, col, self.ai_symbol)
            
            # Update cell styling
            btn = self.buttons[row][col]
            color = COLOR_X if self.ai_symbol == "X" else COLOR_O
            btn.config(text=self.ai_symbol, fg=color, bg=CARD_BG, state="disabled")
            
            if self.check_game_over():
                return
            
            # Hand turn back to human
            self.current_turn = self.human_symbol
            self.status_label.config(text=f"👉 Your Turn ({self.human_symbol})")
            self.set_board_interactive_state(True)

    def set_board_interactive_state(self, enabled):
        """Enables or disables clicking on empty cells."""
        state = "normal" if enabled else "disabled"
        for r in range(3):
            for c in range(3):
                if self.board.get_cell(r, c) == " ":
                    self.buttons[r][c].config(state=state)

    def check_game_over(self):
        """
        Checks backend board state. If game is over, handles popup dialog.
        Returns True if game over, False otherwise.
        """
        winner = self.board.check_winner()
        if winner:
            self.set_board_interactive_state(False)
            if winner == self.human_symbol:
                self.status_label.config(text="🏆 You won!", fg="#4ADE80")
                messagebox.showinfo("Game Over", "Congratulations! You won! 🎉\n(This shouldn't happen with perfect Minimax!)")
            else:
                self.status_label.config(text="💀 AI Wins!", fg="#F87171")
                messagebox.showinfo("Game Over", "AI wins! Better luck next time. 🤖")
            self.root.after(200, self.prompt_replay)
            return True
        elif self.board.is_draw():
            self.set_board_interactive_state(False)
            self.status_label.config(text="🤝 It's a draw!", fg="#FBBF24")
            messagebox.showinfo("Game Over", "It's a draw! Well played! 🤝")
            self.root.after(200, self.prompt_replay)
            return True
        return False

    def prompt_replay(self):
        """Prompts the user to play again or exit."""
        replay = messagebox.askyesno("Play Again?", "Do you want to play another game?")
        if replay:
            self.show_symbol_selection()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

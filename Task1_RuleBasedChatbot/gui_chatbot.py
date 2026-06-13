import datetime
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

# Ensure sys.path includes project directory for imports
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import load_responses, match_rule, clean_input, get_system_time_date

# Colors palette (Discord-like Dark Mode)
BG_COLOR = "#313338"          # Chat area background
HEADER_BG = "#1e1f22"         # Header bar background
INPUT_BG = "#383a40"          # Input area background
TEXT_COLOR = "#dbdee1"        # Default text color
BOT_NAME_COLOR = "#00b0f4"    # Cyan for Bot name
USER_NAME_COLOR = "#23a55a"   # Green for User name
MUTED_TEXT = "#949ba4"        # Muted grey for timestamps and status
ACCENT_COLOR = "#5865f2"      # Discord Blue for buttons
ACCENT_HOVER = "#4752c4"      # Hover state for buttons
BADGE_BG = "#5865f2"          # BOT badge background

class NovaChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nova Chatbot")
        self.root.geometry("480x620")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(400, 500)
        
        # Load chatbot response rules
        self.rules = load_responses()
        
        self.create_widgets()
        self.configure_tags()
        self.display_welcome_message()

    def create_widgets(self):
        # 1. Header Bar
        self.header_frame = tk.Frame(self.root, bg=HEADER_BG, height=60)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)
        self.header_frame.pack_propagate(False)
        
        # Status dot (canvas for drawing online circle)
        self.status_canvas = tk.Canvas(self.header_frame, width=12, height=12, bg=HEADER_BG, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(15, 5))
        self.status_canvas.create_oval(2, 2, 10, 10, fill=USER_NAME_COLOR, outline="")
        
        # Title & Status Labels
        self.title_label = tk.Label(
            self.header_frame, 
            text="Nova Chatbot", 
            font=("Segoe UI", 12, "bold"), 
            bg=HEADER_BG, 
            fg="#ffffff"
        )
        self.title_label.pack(side=tk.LEFT, anchor=tk.W, pady=(8, 0))
        
        self.status_label = tk.Label(
            self.header_frame, 
            text="Active Now", 
            font=("Segoe UI", 9), 
            bg=HEADER_BG, 
            fg=MUTED_TEXT
        )
        # Position labels vertically
        self.title_label.place(x=35, y=10)
        self.status_label.place(x=35, y=30)
        
        # Clear Button
        self.clear_btn = tk.Button(
            self.header_frame,
            text="Clear Chat",
            font=("Segoe UI", 9, "bold"),
            bg="#4e5058",
            fg="#ffffff",
            activebackground="#6d6f78",
            activeforeground="#ffffff",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=5,
            command=self.clear_chat
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=15, pady=15)
        
        # 2. Chat Feed Area
        self.chat_feed = scrolledtext.ScrolledText(
            self.root, 
            bg=BG_COLOR, 
            fg=TEXT_COLOR, 
            font=("Segoe UI", 10), 
            bd=0, 
            highlightthickness=0,
            padx=10,
            pady=10,
            insertbackground="#ffffff" # Cursor color
        )
        self.chat_feed.pack(fill=tk.BOTH, expand=True)
        # Make the chat feed read-only by default
        self.chat_feed.config(state=tk.DISABLED)
        
        # 3. Bottom Input Frame
        self.bottom_frame = tk.Frame(self.root, bg=HEADER_BG, pady=12, padx=15)
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Input entry box
        self.entry_var = tk.StringVar()
        self.entry_box = tk.Entry(
            self.bottom_frame,
            textvariable=self.entry_var,
            bg=INPUT_BG,
            fg=TEXT_COLOR,
            font=("Segoe UI", 10),
            bd=0,
            insertbackground="#ffffff",
            highlightthickness=1,
            highlightcolor=ACCENT_COLOR,
            highlightbackground="#2b2d31"
        )
        self.entry_box.pack(fill=tk.X, side=tk.LEFT, expand=True, ipady=8, padx=(0, 10))
        self.entry_box.focus_set()
        
        # Bind the enter key to send messages
        self.entry_box.bind("<Return>", self.send_message)
        
        # Send Button
        self.send_btn = tk.Button(
            self.bottom_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_COLOR,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=6,
            command=self.send_message
        )
        self.send_btn.pack(side=tk.RIGHT)
        
        # Hover animations for buttons
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.config(bg=ACCENT_HOVER))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.config(bg=ACCENT_COLOR))
        self.clear_btn.bind("<Enter>", lambda e: self.clear_btn.config(bg="#6d6f78"))
        self.clear_btn.bind("<Leave>", lambda e: self.clear_btn.config(bg="#4e5058"))

    def configure_tags(self):
        """Configure font styles and spacing tags in Tkinter Text widget"""
        self.chat_feed.tag_config("user_name", foreground=USER_NAME_COLOR, font=("Segoe UI", 10, "bold"))
        self.chat_feed.tag_config("bot_name", foreground=BOT_NAME_COLOR, font=("Segoe UI", 10, "bold"))
        self.chat_feed.tag_config("bot_badge", background=BADGE_BG, foreground="#ffffff", font=("Segoe UI", 8, "bold"))
        self.chat_feed.tag_config("timestamp", foreground=MUTED_TEXT, font=("Segoe UI", 8))
        self.chat_feed.tag_config("user_text", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        self.chat_feed.tag_config("bot_text", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        self.chat_feed.tag_config("system_text", foreground="#f2a359", font=("Segoe UI", 9, "italic"))
        self.chat_feed.tag_config("spacing", font=("Segoe UI", 4))

    def display_welcome_message(self):
        self.chat_feed.config(state=tk.NORMAL)
        self.chat_feed.insert(
            tk.END, 
            "✨ System: Welcome to Nova Chatbot! I am online and ready to assist.\n"
            "✨ Try typing 'help' to see all available command patterns.\n"
            "--------------------------------------------------------------------------\n", 
            "system_text"
        )
        self.chat_feed.config(state=tk.DISABLED)
        self.chat_feed.see(tk.END)

    def clear_chat(self):
        self.chat_feed.config(state=tk.NORMAL)
        self.chat_feed.delete("1.0", tk.END)
        self.chat_feed.config(state=tk.DISABLED)
        self.display_welcome_message()

    def send_message(self, event=None):
        user_text = self.entry_var.get().strip()
        if not user_text:
            return
            
        # Clear the input entry
        self.entry_var.set("")
        
        # Display user message
        self.append_chat_message("You", user_text, is_user=True)
        
        # Disable input controls while typing effect runs
        self.entry_box.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        
        # Start bot thinking/response in a separate daemon thread
        threading.Thread(target=self.generate_bot_response, args=(user_text,), daemon=True).start()

    def append_chat_message(self, sender, text, is_user=True):
        self.chat_feed.config(state=tk.NORMAL)
        self.chat_feed.insert(tk.END, "\n", "spacing")
        
        if is_user:
            self.chat_feed.insert(tk.END, f"{sender} ", "user_name")
        else:
            self.chat_feed.insert(tk.END, f"{sender} ", "bot_name")
            self.chat_feed.insert(tk.END, " BOT ", "bot_badge")
            
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        self.chat_feed.insert(tk.END, f"  {now_str}\n", "timestamp")
        
        tag = "user_text" if is_user else "bot_text"
        self.chat_feed.insert(tk.END, f"{text}\n", tag)
        self.chat_feed.config(state=tk.DISABLED)
        self.chat_feed.see(tk.END)

    def generate_bot_response(self, user_text):
        # Retrieve the matching rule response
        response = match_rule(user_text, self.rules)
        
        # Add a realistic small delay to simulate processing
        time.sleep(0.4)
        
        # Trigger the typing effect in the GUI thread
        self.root.after(0, self.start_bot_typing, response)

    def start_bot_typing(self, response):
        self.chat_feed.config(state=tk.NORMAL)
        self.chat_feed.insert(tk.END, "\n", "spacing")
        self.chat_feed.insert(tk.END, "Nova ", "bot_name")
        self.chat_feed.insert(tk.END, " BOT ", "bot_badge")
        
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        self.chat_feed.insert(tk.END, f"  {now_str}\n", "timestamp")
        self.chat_feed.config(state=tk.DISABLED)
        self.chat_feed.see(tk.END)
        
        # Begin character-by-character printing
        self.type_character(response, 0)

    def type_character(self, response, index):
        if index < len(response):
            self.chat_feed.config(state=tk.NORMAL)
            self.chat_feed.insert(tk.END, response[index], "bot_text")
            self.chat_feed.config(state=tk.DISABLED)
            self.chat_feed.see(tk.END)
            # Call next character after a brief delay
            self.root.after(10, self.type_character, response, index + 1)
        else:
            # End of message: re-enable input fields
            self.chat_feed.config(state=tk.NORMAL)
            self.chat_feed.insert(tk.END, "\n", "spacing")
            self.chat_feed.config(state=tk.DISABLED)
            
            self.entry_box.config(state=tk.NORMAL)
            self.send_btn.config(state=tk.NORMAL)
            self.entry_box.focus_set()

def main():
    root = tk.Tk()
    app = NovaChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

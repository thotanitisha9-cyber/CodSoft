# Rule-Based Chatbot (Nova)

An interactive, user-friendly, and stylish chatbot built with Python. Named **Nova**, this chatbot matches user inputs with predefined rules and keywords to provide helpful responses, tell jokes, report system date/time, list capabilities, and more. 

Nova offers **two interfaces**:
1. 📟 **CLI (Command Line Interface)**: A colored terminal-based conversation loop.
2. 💻 **GUI (Graphical User Interface)**: A modern, Discord-inspired dark-themed desktop app.

---

## 🚀 Project Overview

Nova is designed to show how rule-based programming can create interactive and user-friendly interfaces. Using pattern matching and keyword dictionaries, Nova offers dynamic responses (randomly selected from multiple variations per category) to keep the conversation engaging.

This project was built as part of the Python programming tasks for the **CodSoft** internship.

---

## ✨ Features

### Core Chatbot Brain
- **Robust Pattern Matching**: Checks for full phrases and single words, ignoring letter casing and common punctuation (e.g. `?`, `!`, `,`).
- **Dynamic Responses**: Selects responses randomly from a collection of replies for each rule, so the chat feels organic.
- **16 Predefined Conversational Rules**:
  1. **Greetings**: `hi`, `hello`, `hey`, `greetings`, `yo`, `hola`
  2. **Identity**: `who are you`, `what is your name`, `what are you`
  3. **How are you**: `how are you`, `how's it going`, `how do you feel`
  4. **Time & Date**: `time`, `date`, `what time is it`, `what is the date`
  5. **Creator**: `who created you`, `who is your maker`, `who developed you`
  6. **Age**: `how old are you`, `your age`, `what is your age`
  7. **Weather**: `weather`, `is it raining`, `what is the weather`
  8. **Jokes**: `tell me a joke`, `joke`, `make me laugh`
  9. **Purpose**: `why do you exist`, `what is your purpose`
  10. **Feelings**: `do you have feelings`, `are you happy`, `are you sad`
  11. **Human Check**: `are you human`, `are you real`, `are you a person`
  12. **Hobbies**: `hobbies`, `what do you do for fun`
  13. **Capabilities**: `what else can you do`, `features`
  14. **Appreciation**: `thank you`, `thanks`, `much appreciated`
  15. **Help/Commands**: `help`, `commands`
  16. **Exit**: `bye`, `exit`, `quit`, `goodbye`

### 📟 CLI Interface Features
- **Terminal Styling**: Beautifully colored CLI using the `colorama` package (user input, bot name, welcome banner, and help instructions are formatted in different colors).
- **Graceful Exit**: Handles commands like `bye`, `exit`, `quit` to end the conversation politely, and catches keyboard interrupts (`Ctrl+C`) to avoid ugly tracebacks.

### 💻 GUI Interface Features
- **Modern Dark Theme**: Styled with a dark-theme chat app layout (bubble-like spacing, custom colors for user and bot, dynamic scrolling).
- **User vs. Bot Layout**: User messages align with green labels, while chatbot messages display with a cyan name and a distinct blue `[BOT]` badge.
- **Asynchronous Typing Simulation**: The bot types out responses character-by-character with a realistic delay, operating on a background thread to keep the interface smooth and responsive.
- **Interactions**:
  - Send messages by clicking the "Send" button or pressing the `Enter` key.
  - Clear conversation history using the "Clear Chat" button in the header.
  - Active status indicator circle (green dot) in the header.

---

## 🛠️ Technologies Used

- **Python 3.7+** (Uses standard libraries `datetime`, `random`, `sys`, `time`, `threading`)
- **Tkinter** (Python's built-in GUI toolkit; no external packages required for the GUI application)
- **Colorama** (`colorama>=0.4.6`) for cross-platform colored terminal output in the CLI mode.

---

## 📦 Installation Steps

1. **Navigate to the project folder**:
   ```bash
   cd RuleBasedChatbot
   ```

2. **(Optional but Recommended) Create a Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   Install the color dependencies (used for the CLI version):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run the Project

You can run either the terminal CLI version or the desktop GUI version:

### 1. Run the GUI Chatbot App (Recommended)
Run the GUI launcher directly:
```bash
python gui_chatbot.py
```

### 2. Run the CLI Chatbot (Terminal-based)
Run the command-line chatbot:
```bash
python chatbot.py
```

---

## 📸 Screenshots

Screenshots showing the CLI interface and the GUI desktop app interface are stored in the [screenshots/](screenshots/) folder:
- **CLI Chatbot Screenshot**: [screenshots/terminal_mockup.png](screenshots/terminal_mockup.png)
- **GUI Chatbot Screenshot**: [screenshots/gui_mockup.png](screenshots/gui_mockup.png)

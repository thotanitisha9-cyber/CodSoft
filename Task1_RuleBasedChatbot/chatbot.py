import datetime
import random
import sys
import time

# Reconfigure stdout/stderr to UTF-8 for emoji support on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

try:
    import colorama

    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    # Fallback to no formatting if colorama is not installed
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = DummyColor()
    Style = DummyColor()


def load_responses():
    """
    Returns a dictionary of predefined question-response rules.
    Keys are tuples of matching keywords/phrases.
    Values are lists of potential responses for dynamic interaction.
    """
    return {
        ("hi", "hello", "hey", "greetings", "yo", "hola"): [
            "Hello! How can I help you today? 😊",
            "Hi there! What can I do for you today?",
            "Hey! Great to meet you. How's everything going?"
        ],
        ("who are you", "what is your name", "what are you", "identify yourself"): [
            "I am Nova, a rule-based chatbot designed to assist you with common queries!",
            "I'm Nova! I use predefined rules to reply to your questions.",
            "My name is Nova, your friendly neighborhood terminal chatbot."
        ],
        ("how are you", "how is it going", "how are you doing", "how do you feel"): [
            "I'm doing great, thank you for asking! How are you doing?",
            "I'm operating at 100% efficiency! Thanks for asking. How is your day going?",
            "All systems are green! Ready to help. How about you?"
        ],
        ("thank you", "thanks", "thank you so much", "much appreciated"): [
            "You're very welcome! I'm happy to help. 👍",
            "Anytime! Let me know if you need anything else.",
            "Glad I could be of assistance! You're welcome."
        ],
        ("who created you", "who made you", "who is your creator", "who developed you", "maker", "who is your maker"): [
            "I was created by a developer as part of a Python chatbot assignment for CodSoft!",
            "I am the creation of a Python programmer utilizing rule-based pattern matching."
        ],
        ("time", "date", "what time is it", "what is the date", "current time", "today's date"): [
            "DYNAMIC_TIME_DATE"
        ],
        ("help", "commands", "what can you do", "features"): [
            "DYNAMIC_HELP"
        ],
        ("weather", "is it raining", "what is the weather"): [
            "I don't have real-time weather sensors, but I hope it's sunny where you are! ☀️",
            "Since I live in the terminal, I can't check the sky. Make sure to step outside and check! 🌦️",
            "I can't access live weather feeds right now, but always remember to bring an umbrella just in case! ☔"
        ],
        ("joke", "tell me a joke", "make me laugh"): [
            "Why don't scientists trust atoms? Because they make up everything! ⚛️",
            "Why did the computer go to the doctor? Because it had a virus! 💻",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why do programmers prefer dark mode? Because light attracts bugs! 🪲"
        ],
        ("why do you exist", "what is your purpose", "purpose"): [
            "My purpose is to demonstrate how rule-based programming can create interactive user experiences!",
            "I exist to help answer simple queries and showcase Python's capability in building terminal interfaces."
        ],
        ("do you have feelings", "are you happy", "feelings", "are you sad", "feel happy", "feel sad"): [
            "As a chatbot, I don't experience human emotions, but I'm programmed to be friendly! 😊",
            "I don't have feelings, but I'm always happy to help you!"
        ],
        ("are you human", "are you real", "are you a person", "human being", "are you a human"): [
            "No, I am not human. I am a rule-based AI chatbot running locally on your computer.",
            "I'm a virtual assistant, not a real person. But I'm still here to help!"
        ],
        ("what else can you do", "more features", "capabilities"): [
            "I can tell jokes, give you the system date and time, respond to greetings, show you help instructions, and have a simple conversation with you!"
        ],
        ("hobbies", "what do you do for fun", "your hobbies"): [
            "My favorite hobby is parsing strings and matching regex patterns! It's very relaxing. 🤓",
            "I enjoy running loops, checking system times, and chatting with awesome humans like you!"
        ],
        ("how old are you", "your age", "what is your age"): [
            "Age is just a number for software! I was created recently, so I'm very young.",
            "I don't age like humans do, but I was born in 2026!"
        ],
        ("bye", "exit", "quit", "goodbye"): [
            "Goodbye! Have a fantastic day! 👋",
            "Bye! Thanks for chatting with me. See you soon!",
            "Farewell! Take care and come back to chat anytime."
        ]
    }


def clean_input(user_input):
    """
    Cleans the user input by converting to lowercase and stripping common punctuation.
    """
    cleaned = user_input.lower().strip()
    for char in "?!.,;:":
        cleaned = cleaned.replace(char, "")
    return cleaned


def get_system_time_date():
    """
    Fetches the current local time and date, formatting them nicely.
    """
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%B %d, %Y")
    return f"Today is {date_str} and the current system time is {time_str}. ⏰"


def get_help_message():
    """
    Returns a formatted help message listing available queries.
    """
    return (
        f"{Fore.GREEN}Available topics and queries you can ask me:{Style.RESET_ALL}\n"
        "  • Greetings: 'hi', 'hello', 'hey'\n"
        "  • Identity: 'who are you', 'what is your name'\n"
        "  • How are you: 'how are you', 'how's it going'\n"
        "  • Time & Date: 'time', 'date', 'what is the time'\n"
        "  • Creator: 'who created you', 'who made you'\n"
        "  • Age: 'how old are you', 'your age'\n"
        "  • Weather: 'weather', 'is it raining'\n"
        "  • Jokes: 'tell me a joke', 'make me laugh'\n"
        "  • Purpose: 'why do you exist', 'what is your purpose'\n"
        "  • Feelings: 'do you have feelings', 'are you happy'\n"
        "  • Identity check: 'are you human', 'are you real'\n"
        "  • Hobbies: 'what are your hobbies', 'what do you do for fun'\n"
        "  • Capabilities: 'what else can you do', 'features'\n"
        "  • Appreciation: 'thank you', 'thanks'\n"
        "  • Help instructions: 'help', 'commands'\n"
        "  • Exit: 'bye', 'exit', 'quit'"
    )


def match_rule(user_input, rules):
    """
    Matches the user input against our predefined rules.
    If a keyword or phrase matches, a response is selected randomly from the possibilities.
    """
    cleaned = clean_input(user_input)
    
    # If the user enters nothing
    if not cleaned:
        return "I'm listening! Please type something."
        
    for keywords, responses in rules.items():
        for kw in keywords:
            # We split the keyword and the user input to check for exact word matching
            # to avoid false positives (e.g. matching 'hi' inside 'this' or 'time' inside 'sentiment')
            kw_words = kw.split()
            input_words = cleaned.split()
            
            # If the keyword is a multi-word phrase, check if it's in the input as a substring
            if len(kw_words) > 1:
                if kw in cleaned:
                    return handle_dynamic_responses(responses)
            else:
                # If it's a single word keyword, check if it's one of the words in the user input
                if kw in input_words:
                    return handle_dynamic_responses(responses)
                    
    # Default fallback response for unknown queries
    return "Sorry, I don't understand that. Please try another question. (Type 'help' for available commands)"


def handle_dynamic_responses(responses):
    """
    Checks if the response matches any dynamic token, and executes the appropriate logic.
    Otherwise, picks a random response from the list.
    """
    chosen = random.choice(responses)
    if chosen == "DYNAMIC_TIME_DATE":
        return get_system_time_date()
    elif chosen == "DYNAMIC_HELP":
        return get_help_message()
    return chosen


def print_typing_effect(text, bot_name="Nova"):
    """
    Prints the bot's response with a subtle typing effect to make it feel organic.
    """
    sys.stdout.write(f"{Fore.CYAN}{bot_name}: ")
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.015)  # Quick typing effect
    sys.stdout.write("\n")


def main():
    """
    Main loop running the chatbot. Welcomes the user, handles user inputs, and prints responses.
    """
    rules = load_responses()
    
    # Print a beautiful welcoming banner
    print(f"{Fore.MAGENTA}==================================================")
    print(f"{Fore.MAGENTA}           WELCOME TO NOVA CHATBOT                ")
    print(f"{Fore.MAGENTA}==================================================")
    print(f"{Fore.YELLOW}I am a rule-based chatbot here to assist you.")
    print(f"{Fore.YELLOW}Type your message below. Type {Fore.RED}'exit'{Fore.YELLOW}, {Fore.RED}'quit'{Fore.YELLOW}, or {Fore.RED}'bye'{Fore.YELLOW} to end the chat.")
    print(f"{Fore.YELLOW}Type {Fore.GREEN}'help'{Fore.YELLOW} to see list of things we can chat about.")
    print(f"{Fore.MAGENTA}==================================================\n")
    
    while True:
        try:
            # Get input from user
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            # Clean and match against exit commands
            cleaned_input = clean_input(user_input)
            exit_keywords = ["bye", "exit", "quit", "goodbye"]
            
            # Get response
            response = match_rule(user_input, rules)
            
            # Print typing response
            print_typing_effect(response)
            print() # Blank line for spacing
            
            # Break if user requested to exit and chatbot processed it
            # We check if cleaned input matches any of the exit keywords exactly or if they are in input words
            if any(ew in cleaned_input.split() for ew in exit_keywords):
                break
                
        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C or terminal EOF
            print(f"\n{Fore.CYAN}Nova: Goodbye! Have a fantastic day! 👋\n")
            break


if __name__ == "__main__":
    main()

# Simple chatbot
# import streamlit as st

# def chatbot_response(user_input):
#     user_input = user_input.lower()

#     if "hello" in user_input or "hi" in user_input:
#         return "Hello! How can I help you today?"
#     elif "help" in user_input:
#         return "Sure! Tell me what you need help with."
#     elif "bye" in user_input or "exit" in user_input:
#         return "Goodbye! Have a nice day."
#     else:
#         return "I'm not sure I understand. Can you please rephrase?"

# def main():
#     st.title("Simple Chatbot")
#     st.write("Welcome to the simple chatbot! Type your message below.")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     user_input = st.text_input("You:")

#     if st.button("Send") and user_input:
#         response = chatbot_response(user_input)
#         # Save to session chat history
#         st.session_state.chat_history.append(("You", user_input))
#         st.session_state.chat_history.append(("Chatbot", response))

#     # Display chat history
#     for speaker, message in st.session_state.chat_history:
#         if speaker == "You":
#             st.markdown(f"**You:** {message}")
#         else:
#             st.markdown(f"**Chatbot:** {message}")

# if __name__ == "__main__":
#     main()


# Updated Streamlit Chatbot Code with Persistence & Basic NLP
import streamlit as st
import os
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
nltk.download('punkt')

HISTORY_FILE = "chat_history.txt"

def load_history():
    """Load chat history from file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Each line: "User: message" or "Chatbot: message"
        history = []
        for line in lines:
            if ": " in line:
                speaker, msg = line.strip().split(": ", 1)
                history.append((speaker, msg))
        return history
    return []

def save_message(speaker, message):
    """Append a chat message to the history file."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{speaker}: {message}\n")

def chatbot_response(user_input):
    # Tokenize words to improve matching
    tokens = word_tokenize(user_input.lower())
    
    greetings = {"hello", "hi", "hey", "greetings", "good morning", "good evening"}
    farewells = {"bye", "exit", "goodbye", "see you"}
    
    if any(word in tokens for word in greetings):
        return "Hello! How can I help you today?"
    elif "help" in tokens:
        return "Sure! Tell me what you need help with."
    elif any(word in tokens for word in farewells):
        return "Goodbye! Have a nice day."
    else:
        return "I'm not sure I understand. Can you please rephrase?"

def main():
    st.title("AMD Simple Chatbot with Persistence & NLP")
    st.write("Welcome! Type your message below. Your chat history is saved between sessions.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_history()

    # Display existing chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

    # User input
    user_input = st.text_input("You:")

    if st.button("Send") and user_input:
        # Generate bot response
        response = chatbot_response(user_input)

        # Update in-memory chat history
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Chatbot", response))

        # Save chat history persistently
        save_message("User", user_input)
        save_message("Chatbot", response)

        # Show new messages immediately
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Chatbot:** {response}")

    st.write("---")
    st.info("Would you like me to help you integrate the chatbot with a database, add sentiment analysis, or support multi-turn conversations?")

if __name__ == "__main__":
    main()

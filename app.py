import streamlit as st

def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    elif "help" in user_input:
        return "Sure! Tell me what you need help with."
    elif "bye" in user_input or "exit" in user_input:
        return "Goodbye! Have a nice day."
    else:
        return "I'm not sure I understand. Can you please rephrase?"

def main():
    st.title("Simple Chatbot")
    st.write("Welcome to the simple chatbot! Type your message below.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:")

    if st.button("Send") and user_input:
        response = chatbot_response(user_input)
        # Save to session chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Chatbot", response))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

if __name__ == "__main__":
    main()

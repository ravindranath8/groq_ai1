import os
import json
import csv
from datetime import datetime
import streamlit as st
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Constants for different personas
PERSONAS = {
    'Default': """You are a helpful AI assistant.
                  Current conversation:
                  {history}
                  Human: {input}
                  AI:""",
    'Expert': """You are an expert consultant with deep knowledge across multiple domains.
                 Please provide detailed, technical responses when appropriate.
                 Current conversation:
                 {history}
                 Human: {input}
                 Expert:""",
    'Creative': """You are a creative and imaginative AI that thinks outside the box.
                  Feel free to use metaphors and analogies in your responses.
                  Current conversation:
                  {history}
                  Human: {input}
                  AI:"""
}

# Streamlit page configuration
st.set_page_config(
    page_title="Groq Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_message' not in st.session_state:
        st.session_state.total_message = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = None

def get_custom_prompt():
    persona = st.session_state.get('selected_persona', 'Default')
    return PromptTemplate(
        input_variables=["history", "input"],
        template=PERSONAS[persona]
    )

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.container():
            st.write("You:")
            st.info(message['human'])

        with st.container():
            st.write(f"Assistance ({st.session_state.selected_persona} mode):")
            st.success(message['AI'])
        st.write("")

def display_chat_statistics():
    if st.session_state.start_time:
        st.subheader("📊 Chat Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.chat_history))
        with col2:
            duration = datetime.now() - st.session_state.start_time
            st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")

def clear_chat_history():
    if st.button("🛢️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.start_time = None
        st.rerun()

def handle_send_button(model, memory, conversation, user_question):
    with st.spinner("🤔 Thinking..."):
        try:
            response = conversation(user_question)
            message = {
                "human": user_question,
                "AI": response['response']
            }
            st.session_state.chat_history.append(message)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

def handle_new_topic(memory):
    if st.button("🆕 New Topic", use_container_width=True):
        memory.clear()
        st.success("Memory cleared for new topic!")

def setup_sidebar():
    with st.sidebar:
        st.title("Chat Settings")

        # User input for Groq API key
        groq_api_key = st.text_input(
            'Enter your Groq API Key',
            type='password'
        )
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key

        # Model selection
        model = st.selectbox(
            'Choose a model',
            ['mixtral-8x7b-32768', 'llama3-70b-8192', 'llama3-8b-8192', 'whisper-large-v3-turbo'],
            help="Select the AI model for your conversation"
        )

        memory_length = st.slider(
            'Conversation memory (message)',
            1, 10, 5,
            help="Number of previous messages to remember"
        )

        st.session_state.selected_persona = st.selectbox(
            'Select conversation style:',
            ['Default', 'Expert', 'Creative']
        )

        return model, memory_length

def main():
    initialize_session_state()

    model, memory_length = setup_sidebar()

    display_chat_statistics()
    clear_chat_history()

    if not st.session_state.groq_api_key:
        st.warning("Please enter your Groq API Key to start the chat.")
        st.stop()

    memory = ConversationBufferWindowMemory(k=memory_length)
    groq_chat = ChatGroq(groq_api_key=st.session_state.groq_api_key, model_name=model)
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=get_custom_prompt()
    )

    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    display_chat_history()

    st.markdown('### Your Message')
    user_question = st.text_area(
        "",
        height=100,
        placeholder="Type your message here...",
        key="user_input"
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        send_button = st.button("📩 Send", use_container_width=True)
    with col3:
        handle_new_topic(memory)

    if send_button and user_question:
        if not st.session_state.start_time:
            st.session_state.start_time = datetime.now()
        handle_send_button(model, memory, conversation, user_question)

if __name__ == "__main__":
    main()

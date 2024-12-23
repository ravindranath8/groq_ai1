import os
from datetime import datetime
import streamlit as st
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
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

# Initialize the Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Streamlit page configuration
st.set_page_config(
    page_title="Groq chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_message' not in st.session_state:
        st.session_state.total_message = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

def get_custom_prompt():
    """Return the prompt template based on selected persona."""
    persona = st.session_state.get('selected_persona', 'Default')
    return PromptTemplate(
        input_variables=["history", "input"],
        template=PERSONAS[persona]
    )

def display_chat_history():
    """Display the conversation history."""
    for message in st.session_state.chat_history:
        with st.container():
            st.write("You:")
            st.info(message['human'])

        with st.container():
            st.write(f"Assistance ({st.session_state.selected_persona} mode):")
            st.success(message['AI'])
        st.write("")

def display_chat_statistics():
    """Display chat statistics if available."""
    if st.session_state.start_time:
        st.subheader("📊 Chat Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.chat_history))
        with col2:
            duration = datetime.now() - st.session_state.start_time
            st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")

def clear_chat_history():
    """Clear the chat history and restart the session."""
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.start_time = None
        st.rerun()

def handle_send_button(model, memory, conversation, user_question):
    """Handle sending a message and getting a response."""
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
    """Handle clearing the memory for a new topic."""
    if st.button("🆕 New Topic", use_container_width=True):
        memory.clear()
        st.success("Memory cleared for new topic!")

def setup_sidebar():
    """Setup the sidebar with model, memory, and persona selection."""
    with st.sidebar:
        st.title("Chat Settings")

        # Model selection
        model = st.selectbox(
            'Choose a model',
            ['mixtral-8x7b-32768', 'llama3-70b-8192', 'llama3-8b-8192'],
            help="Select the AI model for your conversation"
        )

        # Memory setting
        memory_length = st.slider(
            'Conversation memory (message)',
            1, 10, 5,
            help="Number of previous messages to remember"
        )

        # AI Persona
        st.session_state.selected_persona = st.selectbox(
            'Select conversation style:',
            ['Default', 'Expert', 'Creative']
        )

        return model, memory_length

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    model, memory_length = setup_sidebar()

    # Chat statistics and clear history
    display_chat_statistics()
    clear_chat_history()

    # Initialize memory and conversation chain
    memory = ConversationBufferWindowMemory(k=memory_length)
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=get_custom_prompt()
    )

    # Save chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    # Display chat history
    display_chat_history()

    # User input
    st.markdown('### Your Message')
    user_question = st.text_area(
        "",
        height=100,
        placeholder="Type your message here...(Shift + Enter to send)",
        key="user_input",
        help="Type your message and press Shift + Enter or click the send button"
    )

    # Action buttons (Send and New Topic)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        send_button = st.button("📩 Send", use_container_width=True)
    with col3:
        handle_new_topic(memory)

    # Handle the send button click
    if send_button and user_question:
        if not st.session_state.start_time:
            st.session_state.start_time = datetime.now()
        handle_send_button(model, memory, conversation, user_question)

    # Footer
    st.markdown("---")
    st.markdown(
        f"Using Groq AI with {st.session_state.selected_persona.lower()} persona | "
        f"Memory: {memory_length} messages"
    )

if __name__ == "__main__":
    main()

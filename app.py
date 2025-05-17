import streamlit as st
import os
from lead_agent.lead_agent import LeadAgent

# Function to get available books with valid FAISS indexes
def get_available_books():
    books = []
    # Check if faiss_indexes directory exists
    if not os.path.exists("faiss_indexes"):
        return books
    
    # Look for directories that contain both index.faiss and index.pkl files
    for folder in os.listdir("faiss_indexes"):
        folder_path = os.path.join("faiss_indexes", folder)
        if os.path.isdir(folder_path):
            if (os.path.exists(os.path.join(folder_path, "index.faiss")) and 
                os.path.exists(os.path.join(folder_path, "index.pkl"))):
                # Convert folder name back to book title format (replace underscores with spaces and capitalize)
                book_title = " ".join(word.capitalize() for word in folder.split("_"))
                books.append(book_title)
    return books

# Initialize the LeadAgent
lead_agent = LeadAgent()

# Set up the Streamlit app
st.set_page_config(page_title="BookHub – AI Book Companion", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 800px;
        margin: 0 auto;
    }
    .stApp {
        background-color: #f9f9f9;
    }
    .chat-container {
        max-height: 550px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .user-bubble {
        background-color: #e1f5fe;
        color: #0277bd;
        border-radius: 18px 18px 0 18px;
        padding: 10px 15px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    .bot-bubble {
        background-color: #f1f1f1;
        color: #424242;
        border-radius: 18px 18px 18px 0;
        padding: 10px 15px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
        word-wrap: break-word;
    }
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    .chat-footer {
        margin-top: 10px;
    }
    .stButton button {
        background-color: #0277bd;
        color: white;
        border-radius: 20px;
    }
    .stTextInput input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📚 BookHub – AI Book Companion</h1>", unsafe_allow_html=True)

# Get available books
available_books = get_available_books()

# Initialize session state for message history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history in a scrollable container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state['messages']:
        if message['type'] == 'user':
            st.markdown(f'<div class="user-bubble">{message["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">{message["content"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Create a form for user input with Send button
with st.form(key='chat_form', clear_on_submit=True):
    # Book selection dropdown
    if available_books:
        selected_book = st.selectbox("Select a book:", available_books)
        placeholder_text = "Ask anything about this book..."
    else:
        selected_book = None
        placeholder_text = "No books available with FAISS indexes. Please add a book first."
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Your message", placeholder=placeholder_text, label_visibility="collapsed", disabled=not available_books)
    with col2:
        submit_button = st.form_submit_button("Send", disabled=not available_books)

# Handle user input
if submit_button and user_input and selected_book:
    # Add user message to session state
    st.session_state['messages'].append({'type': 'user', 'content': user_input})

    # Get response from LeadAgent's route method
    response = lead_agent.route(user_input, selected_book)

    # Add AI response to session state
    st.session_state['messages'].append({'type': 'bot', 'content': response})

    # Force rerun to refresh the display
    st.rerun() 
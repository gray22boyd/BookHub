import streamlit as st
import os
from config.settings import config
from services.BookApiClient import OrganizingAgent

# Page configuration
st.set_page_config(
    page_title="BookHub AI Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .book-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .usage-tip {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 BookHub AI Book Companion</h1>
        <p>Your intelligent assistant for exploring and understanding books</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agents
    lead_agent = LeadAgent()
    organizer_agent = OrganizingAgent()
    
    # Sidebar
    with st.sidebar:
        st.header("📖 Book Management")
        
        # Available books
        faiss_dir = config.FAISS_INDEXES_DIR
        available_books = []
        if os.path.exists(faiss_dir):
            available_books = [d for d in os.listdir(faiss_dir) 
                             if os.path.isdir(os.path.join(faiss_dir, d))]
        
        if available_books:
            st.subheader("Available Books:")
            for book in available_books:
                book_display = book.replace('_', ' ').title()
                st.markdown(f"<div class='book-card'>📚 {book_display}</div>", 
                           unsafe_allow_html=True)
        else:
            st.info("No books available. Add some books using the section below!")
        
        st.divider()
        
        # Add new book
        st.subheader("➕ Add New Book")
        
        with st.expander("Add Book from Project Gutenberg"):
            book_id = st.number_input("Enter Gutenberg Book ID:", 
                                    min_value=1, value=11, step=1)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Book", key="add_book"):
                    with st.spinner("📥 Looking up book..."):
                        try:
                            # First get book info to get the real title
                            import requests
                            response = requests.get(f"https://gutendx.com/books/{book_id}")
                            if response.status_code == 200:
                                book_data = response.json()
                                book_title = book_data.get('title', f'Book ID {book_id}')
                                with st.spinner("📥 Downloading and processing book..."):
                                    result = organizer_agent.add_book(book_title)
                                    if "successfully" in result.lower():
                                        st.success(f"✅ {result}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {result}")
                            else:
                                st.error(f"❌ Could not find book with ID {book_id}")
                        except Exception as e:
                            st.error(f"❌ Error looking up book: {str(e)}")
            
            with col2:
                if st.button("Remove Book", key="remove_book"):
                    if available_books:
                        book_to_remove = st.selectbox("Select book to remove:", 
                                                    available_books, key="book_select")
                        if book_to_remove:
                            st.info("📝 Book removal feature not yet implemented. You can manually delete the book folder from faiss_indexes directory.")
                    else:
                        st.info("No books to remove")
        
        # Popular books suggestions
        st.markdown("""
        <div class="usage-tip">
            <strong>📋 Popular Book IDs:</strong><br>
            • Alice in Wonderland: 11<br>
            • Pride and Prejudice: 1342<br>
            • Frankenstein: 84<br>
            • Dracula: 345<br>
            • The Great Gatsby: 64317
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Your Books")
        
        # Book selection for chat
        if available_books:
            # Create a mapping of display names to directory names
            book_options = {}
            for book_dir in available_books:
                display_name = book_dir.replace('_', ' ').title()
                book_options[display_name] = book_dir
            
            selected_display = st.selectbox(
                "📖 Select a book to chat about:",
                options=list(book_options.keys()),
                key="chat_book_selection"
            )
            selected_book_dir = book_options[selected_display]
        else:
            selected_display = None
            selected_book_dir = None
            st.warning("⚠️ No books available. Please add a book first using the sidebar.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your books..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Pass the directory name directly to match what AnswerAgent expects
                        if selected_book_dir:
                            response = lead_agent.handle_prompt(prompt, selected_book_dir)
                        else:
                            response = "Please select a book first before asking questions."
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.header("💡 Usage Tips")
        
        st.markdown("""
        <div class="usage-tip">
            <strong>🎯 Try these queries:</strong><br><br>
            
            <strong>Specific Content:</strong><br>
            • "What is the 15th sentence in Alice in Wonderland?"<br>
            • "Find the 3rd paragraph in Chapter 1 of Pride and Prejudice"<br><br>
            
            <strong>Analysis:</strong><br>
            • "Summarize the main themes in Alice in Wonderland"<br>
            • "Analyze the character development in Pride and Prejudice"<br><br>
            
            <strong>Search:</strong><br>
            • "Find passages about the Cheshire Cat"<br>
            • "What does the book say about love?"
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # System status
        st.header("📊 System Status")
        
        status_container = st.container()
        with status_container:
            # Check if OpenAI API key is set
            if config.OPENAI_API_KEY:
                st.success("✅ OpenAI API Key configured")
            else:
                st.warning("⚠️ OpenAI API Key not set")
            
            # Check embedding model
            st.info(f"🧠 Embedding Model: {config.EMBEDDING_MODEL}")
            
            # Check available books count
            book_count = len(available_books)
            if book_count > 0:
                st.success(f"📚 {book_count} book(s) available")
            else:
                st.warning("📚 No books loaded")

if __name__ == "__main__":
    main() 
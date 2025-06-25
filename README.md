# 📚 BookHub AI Book Companion

An intelligent Streamlit application that allows you to chat with books using advanced AI and vector embeddings. Ask specific questions about book content, get summaries, analyze themes, and more!

## ✨ Features

- **Smart Book Chat**: Ask questions about specific sentences, paragraphs, themes, and characters
- **Advanced Embeddings**: Uses `intfloat/e5-large-v2` for superior semantic understanding
- **Project Gutenberg Integration**: Automatically download and process books from Project Gutenberg
- **Modern UI**: Beautiful, responsive interface with chat history
- **Specific Content Retrieval**: Find exact sentences and paragraphs by number
- **Book Management**: Easy book addition and selection

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gray22boyd/BookHub.git
   cd BookHub
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎯 Usage Examples

### Adding Books
- Enter a Project Gutenberg book ID (e.g., 11 for Alice in Wonderland)
- Click "Add Book" and wait for processing
- Popular book IDs are suggested in the sidebar

### Asking Questions
- **Specific Content**: "What is the 15th sentence in Alice in Wonderland?"
- **Analysis**: "Summarize the main themes in Pride and Prejudice"
- **Character Questions**: "Tell me about the Cheshire Cat"
- **Search**: "Find passages about love"

## 🏗️ Architecture

### Core Components

- **`app.py`**: Main Streamlit interface with modern UI
- **`lead_agent/`**: Routing agent that directs queries to appropriate handlers
- **`agents/`**: Specialized agents for book Q&A and organization
- **`config/`**: Centralized configuration management
- **`services/`**: Shared services like embedding model management

### Key Features

- **Centralized Configuration**: All settings managed in `config/settings.py`
- **Singleton Embedding Service**: Prevents duplicate model loading
- **Smart Query Routing**: Automatically routes queries to appropriate agents
- **Consistent Book Handling**: Proper mapping between UI names and directory names

## 🛠️ Technical Details

### Embeddings
- **Model**: `intfloat/e5-large-v2` (1024 dimensions)
- **Chunk Size**: 400 tokens with 50 token overlap
- **Vector Store**: FAISS for efficient similarity search

### Dependencies
- **Streamlit**: Modern web interface
- **LangChain**: LLM orchestration and document processing
- **OpenAI**: GPT-4 for question answering
- **Sentence Transformers**: E5 embedding model
- **FAISS**: Vector similarity search
- **tiktoken**: Token counting and text chunking

## 📁 Project Structure

```
BookHub/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
├── config/
│   ├── __init__.py
│   └── settings.py      # Centralized configuration
├── agents/
│   ├── answer_agent.py  # Book Q&A handler
│   └── organizer_agent.py # Book management
├── lead_agent/
│   └── lead_agent.py    # Query routing
├── services/
│   ├── __init__.py
│   └── embedding_service.py # Shared embedding model
└── faiss_indexes/       # Generated book indexes (excluded from git)
    ├── alice_and_wonderland/
    └── pride_and_prejudice/
```

## 🔧 Configuration

All configuration is centralized in `config/settings.py`:

- **API Keys**: OpenAI API configuration
- **Embedding Settings**: Model selection and dimensions
- **Text Processing**: Chunk sizes and overlap
- **Retrieval**: Search parameters
- **Paths**: Directory locations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Project Gutenberg** for providing free access to classic literature
- **Hugging Face** for the excellent E5 embedding model
- **OpenAI** for GPT-4 language model
- **LangChain** for the document processing framework

## 🐛 Troubleshooting

### Common Issues

1. **"No index found" error**: Make sure you've added books using the sidebar
2. **OpenAI API errors**: Check your API key in the `.env` file
3. **Memory issues**: The E5 model requires ~2GB RAM for optimal performance
4. **Slow responses**: First query may be slower due to model loading

### Getting Help

- Check the [Issues](https://github.com/gray22boyd/BookHub/issues) page
- Create a new issue with detailed error information
- Include your Python version and operating system

---

**Happy Reading! 📖✨** 
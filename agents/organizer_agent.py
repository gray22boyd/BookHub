import os
import re
import requests
import tiktoken
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config.settings import config
from services.embedding_service import EmbeddingService

class OrganizingAgent:
    """
    Agent responsible for downloading, processing, and indexing books.
    Uses centralized configuration and shared embedding service.
    """
    
    def __init__(self):
        """Initialize the organizing agent."""
        self.embedding_service = EmbeddingService()
        
    def _sanitize_title(self, title: str) -> str:
        """Sanitize book title for use as directory name."""
        return re.sub(r'[^\w\s-]', '', title.lower()).replace(' ', '_')
    
    def locate_book(self, book_title: str) -> str:
        """
        Find and download a plain text version of a book from the Gutenberg API.
        
        Args:
            book_title: Title of the book to search for
            
        Returns:
            Full text content of the book
            
        Raises:
            Exception: If book cannot be found or downloaded
        """
        try:
            # Search for the book
            search_url = f"https://gutendx.com/books/?search={book_title}"
            response = requests.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                raise ValueError(f"No books found for '{book_title}' on Project Gutenberg")
                
            book = data["results"][0]
            formats = book.get("formats", {})
            
            # Try multiple text formats in order of preference
            text_url = None
            format_preferences = [
                "text/plain; charset=utf-8",
                "text/plain; charset=us-ascii", 
                "text/plain"
            ]
            
            for format_key in format_preferences:
                if format_key in formats:
                    text_url = formats[format_key]
                    break
                    
            if not text_url:
                available_formats = list(formats.keys())
                raise ValueError(
                    f"No plain text format available for '{book_title}'. "
                    f"Available formats: {', '.join(available_formats)}"
                )
            
            # Download the book text
            print(f"Downloading '{book_title}' from {text_url}")
            text_response = requests.get(text_url, timeout=60)
            text_response.raise_for_status()
            
            if not text_response.text:
                raise ValueError(f"Downloaded text is empty for '{book_title}'")
                
            print(f"Successfully downloaded '{book_title}' ({len(text_response.text)} characters)")
            return text_response.text
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download book '{book_title}': {e}")
        except Exception as e:
            raise Exception(f"Error processing book '{book_title}': {e}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using tiktoken tokenizer.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks with passage prefixes
        """
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Normalize newlines for consistent splitting
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove Project Gutenberg header/footer if present
            text = self._clean_gutenberg_text(text)
            
            # Split into paragraphs for semantic boundaries
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            chunks = []
            current_chunk = []
            current_token_count = 0
            
            for para in paragraphs:
                tokens = tokenizer.encode(para)
                
                # If adding this paragraph exceeds the limit, finalize current chunk
                if current_token_count + len(tokens) > config.CHUNK_SIZE:
                    if current_chunk:  # Only add non-empty chunks
                        full_chunk = " ".join(current_chunk)
                        chunks.append(full_chunk)
                    
                    # Start new chunk with overlap if configured
                    if config.CHUNK_OVERLAP > 0 and current_chunk:
                        # Take the last part for overlap
                        previous_text = " ".join(current_chunk)
                        prev_tokens = tokenizer.encode(previous_text)
                        if len(prev_tokens) > config.CHUNK_OVERLAP:
                            overlap_tokens = prev_tokens[-config.CHUNK_OVERLAP:]
                            overlap_text = tokenizer.decode(overlap_tokens)
                            current_chunk = [overlap_text]
                            current_token_count = len(overlap_tokens)
                        else:
                            current_chunk = []
                            current_token_count = 0
                    else:
                        current_chunk = []
                        current_token_count = 0
                
                current_chunk.append(para)
                current_token_count += len(tokens)
            
            # Add any remaining chunk
            if current_chunk:
                full_chunk = " ".join(current_chunk)
                chunks.append(full_chunk)
            
            # Add passage prefixes as required by e5 model
            prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
            
            print(f"Created {len(prefixed_chunks)} chunks from text")
            return prefixed_chunks
            
        except Exception as e:
            raise Exception(f"Failed to chunk text: {e}")
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Project Gutenberg header and footer."""
        # Remove header (everything before "*** START OF")
        start_match = re.search(r'\*\*\*\s*START OF (THE )?PROJECT GUTENBERG', text, re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]
            # Remove the rest of the start line
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline + 1:]
        
        # Remove footer (everything after "*** END OF")
        end_match = re.search(r'\*\*\*\s*END OF (THE )?PROJECT GUTENBERG', text, re.IGNORECASE)
        if end_match:
            text = text[:end_match.start()]
        
        return text.strip()
    
    def add_book(self, book_title: str) -> str:
        """
        Add a book to the system by downloading, chunking, and indexing it.
        
        Args:
            book_title: Title of the book to add
            
        Returns:
            Success or error message
        """
        if not book_title or not book_title.strip():
            return " Please provide a book title."
        
        book_title = book_title.strip()
        sanitized_title = self._sanitize_title(book_title)
        
        try:
            # Check if book already exists
            book_folder = os.path.join(config.FAISS_INDEXES_DIR, sanitized_title)
            if os.path.exists(book_folder):
                return f" Book '{book_title}' already exists in the system!"
            
            # Create directories
            os.makedirs(config.FAISS_INDEXES_DIR, exist_ok=True)
            os.makedirs(book_folder, exist_ok=True)
            
            print(f"Processing '{book_title}'...")
            
            # Step 1: Download book
            book_text = self.locate_book(book_title)
            
            # Step 2: Chunk the text
            chunks = self.chunk_text(book_text)
            
            if not chunks:
                raise ValueError("No text chunks were created from the book")
            
            # Step 3: Create documents for LangChain
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Step 4: Get embeddings and create vector store
            print(f"Creating embeddings for {len(documents)} chunks...")
            embeddings = self.embedding_service.get_embeddings()
            
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Step 5: Save to disk
            print(f"Saving index to {book_folder}")
            vectorstore.save_local(book_folder)
            
            return f" Successfully added '{book_title}' to the system! You can now ask questions about it."
            
        except Exception as e:
            # Clean up partial files if error occurred
            try:
                if os.path.exists(book_folder):
                    import shutil
                    shutil.rmtree(book_folder)
            except:
                pass
            
            error_msg = str(e)
            if "No books found" in error_msg:
                return f" Could not find '{book_title}' on Project Gutenberg. Please check the title and try again."
            elif "No plain text format" in error_msg:
                return f" '{book_title}' is not available in plain text format on Project Gutenberg."
            elif "Failed to download" in error_msg:
                return f" Failed to download '{book_title}'. Please check your internet connection and try again."
            else:
                return f" Error adding book '{book_title}': {error_msg}"
    
    def get_available_books(self) -> List[str]:
        """Get list of books currently available in the system."""
        try:
            if not os.path.exists(config.FAISS_INDEXES_DIR):
                return []
            
            books = []
            for name in os.listdir(config.FAISS_INDEXES_DIR):
                path = os.path.join(config.FAISS_INDEXES_DIR, name)
                if os.path.isdir(path):
                    # Convert sanitized name back to readable format
                    readable_name = name.replace('_', ' ').title()
                    books.append(readable_name)
            
            return sorted(books)
        except Exception:
            return []
    
    def handle(self, prompt: str) -> str:
        """
        Handle organization requests like adding books.
        
        Args:
            prompt: User's request
            
        Returns:
            Response message
        """
        if not prompt or not prompt.strip():
            return " Please provide a request. You can add books by saying 'add [book title]'."
        
        prompt = prompt.strip().lower()
        
        try:
            # Extract book title from prompt
            if any(keyword in prompt for keyword in ["add", "upload", "ingest"]):
                # Find the action word and extract everything after it
                for keyword in ["add", "upload", "ingest"]:
                    if keyword in prompt:
                        # Split on the keyword and take everything after
                        parts = prompt.split(keyword, 1)
                        if len(parts) > 1 and parts[1].strip():
                            book_title = parts[1].strip()
                            # Remove common words that might be included
                            book_title = re.sub(r'^(book|the book|a book)\s+', '', book_title, flags=re.IGNORECASE)
                            return self.add_book(book_title)
                        break
                
                return " Please specify a book title. Example: 'add Pride and Prejudice'"
            
            elif "list" in prompt or "show" in prompt or "available" in prompt:
                books = self.get_available_books()
                if books:
                    return f" Available books:\n" + "\n".join(f" {book}" for book in books)
                else:
                    return " No books have been added to the system yet."
            
            else:
                return " I can help you add books to the system. Try: 'add [book title]' or 'list books'."
                
        except Exception as e:
            return f" An unexpected error occurred: {str(e)}"

import os
import re
import requests
import tiktoken
import nltk
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config.settings import config
from services.embedding_service import EmbeddingService

# Download NLTK data on first import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
        Split text into overlapping chunks using sentence-aware chunking with tiktoken.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks with passage prefixes
        """
        try:
            from nltk.tokenize import sent_tokenize
            
            tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Normalize newlines for consistent splitting
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove Project Gutenberg header/footer if present
            text = self._clean_gutenberg_text(text)
            
            # Use NLTK for better sentence tokenization
            sentences = sent_tokenize(text)
            
            chunks = []
            current_chunk = ""
            current_tokens = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                tokens = tokenizer.encode(sentence)
                
                # If adding this sentence exceeds the limit, finalize current chunk
                if current_tokens + len(tokens) > config.CHUNK_SIZE:
                    if current_chunk:  # Only add non-empty chunks
                        chunks.append(f"passage: {current_chunk.strip()}")
                    
                    # Handle overlap - take last few sentences from current chunk
                    if config.CHUNK_OVERLAP > 0 and current_chunk:
                        # Split current chunk into sentences for overlap
                        chunk_sentences = sent_tokenize(current_chunk)
                        overlap_sentences = []
                        overlap_tokens = 0
                        
                        # Take sentences from the end until we reach overlap limit
                        for overlap_sentence in reversed(chunk_sentences):
                            overlap_sentence_tokens = len(tokenizer.encode(overlap_sentence))
                            if overlap_tokens + overlap_sentence_tokens <= config.CHUNK_OVERLAP:
                                overlap_sentences.insert(0, overlap_sentence)
                                overlap_tokens += overlap_sentence_tokens
                            else:
                                break
                        
                        if overlap_sentences:
                            current_chunk = " ".join(overlap_sentences)
                            current_tokens = overlap_tokens
                        else:
                            current_chunk = ""
                            current_tokens = 0
                    else:
                        current_chunk = ""
                        current_tokens = 0
                
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += len(tokens)
            
            # Add any remaining chunk
            if current_chunk:
                chunks.append(f"passage: {current_chunk.strip()}")
            
            print(f"Created {len(chunks)} sentence-aware chunks from text")
            return chunks
            
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
            
            # Step 2: Clean the text
            cleaned_text = self._clean_gutenberg_text(book_text)
            
            # Step 2.5: Save full cleaned book text for accurate indexing
            text_path = os.path.join(book_folder, "book_text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved full cleaned text to {text_path}")
            
            # Step 3: Chunk the text
            chunks = self.chunk_text(cleaned_text)
            
            if not chunks:
                raise ValueError("No text chunks were created from the book")
            
            # Step 4: Create documents for LangChain
            documents = [Document(page_content=chunk) for chunk in chunks]
            texts = [doc.page_content for doc in documents]
            
            # Progress print for large books
            batch_size = 50
            print(f"Starting embedding in batches of {batch_size}...")
            embeddings = self.embedding_service.get_embeddings()
            all_vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"Embedding batch {i//batch_size+1} ({i}-{min(i+batch_size, len(texts))} of {len(texts)})")
                try:
                    vectors = embeddings.embed_documents(batch)
                    all_vectors.extend(vectors)
                except Exception as e:
                    print(f"❌ Embedding failed for batch {i//batch_size+1}: {e}")
                    raise
            print("Embedding complete.")
            
            # Step 6: Save to disk
            print(f"Saving index to {book_folder}")
            try:
                vectorstore = FAISS.from_embeddings(list(zip(all_vectors, texts)))
                vectorstore.save_local(book_folder)
                print("Index saved successfully.")
            except Exception as e:
                print(f"❌ Failed to save FAISS index: {e}")
                raise
            
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
    
    def add_book_from_file(self, book_title: str, file_path: str) -> str:
        """
        Add a book from a local text file when Gutenberg API is unavailable.
        
        Args:
            book_title: Title of the book
            file_path: Path to the text file
            
        Returns:
            Success or error message
        """
        if not book_title or not book_title.strip():
            return " Please provide a book title."
        
        if not os.path.exists(file_path):
            return f" File not found: {file_path}"
        
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
            
            print(f"Processing '{book_title}' from local file...")
            
            # Step 1: Read book from file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    book_text = f.read()
            except UnicodeDecodeError:
                # Try different encodings
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        book_text = f.read()
                except:
                    with open(file_path, 'r', encoding='cp1252') as f:
                        book_text = f.read()
            
            if not book_text or len(book_text) < 100:
                raise ValueError("File appears to be empty or too short")
            
            # Step 2: Clean the text
            cleaned_text = self._clean_gutenberg_text(book_text)
            
            # Step 2.5: Save full cleaned book text for accurate indexing
            text_path = os.path.join(book_folder, "book_text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved full cleaned text to {text_path}")
            
            # Step 3: Chunk the text
            chunks = self.chunk_text(cleaned_text)
            
            if not chunks:
                raise ValueError("No text chunks were created from the book")
            
            # Step 4: Create documents for LangChain
            documents = [Document(page_content=chunk) for chunk in chunks]
            texts = [doc.page_content for doc in documents]
            
            # Progress print for large books
            batch_size = 50
            print(f"Starting embedding in batches of {batch_size}...")
            embeddings = self.embedding_service.get_embeddings()
            all_vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"Embedding batch {i//batch_size+1} ({i}-{min(i+batch_size, len(texts))} of {len(texts)})")
                try:
                    vectors = embeddings.embed_documents(batch)
                    all_vectors.extend(vectors)
                except Exception as e:
                    print(f"❌ Embedding failed for batch {i//batch_size+1}: {e}")
                    raise
            print("Embedding complete.")
            
            # Step 6: Save to disk
            print(f"Saving index to {book_folder}")
            vectorstore = FAISS.from_embeddings(all_vectors, documents)
            vectorstore.save_local(book_folder)
            
            return f" Successfully added '{book_title}' from local file! You can now ask questions about it."
            
        except Exception as e:
            # Clean up partial files if error occurred
            try:
                if os.path.exists(book_folder):
                    import shutil
                    shutil.rmtree(book_folder)
            except:
                pass
            
            return f" Error adding book '{book_title}' from file: {str(e)}"

    def download_sample_books(self) -> str:
        """
        Download sample books from alternative sources when Gutenberg is down.
        """
        sample_books = {
            "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "Frankenstein": "https://www.gutenberg.org/files/84/84-0.txt", 
            "Alice's Adventures in Wonderland": "https://www.gutenberg.org/files/11/11-0.txt"
        }
        
        results = []
        for title, url in sample_books.items():
            try:
                print(f"Trying to download {title} from gutenberg.org...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                if response.text and len(response.text) > 1000:
                    # Save to temporary file
                    sanitized = self._sanitize_title(title)
                    temp_file = f"temp_{sanitized}.txt"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    # Add using file method
                    result = self.add_book_from_file(title, temp_file)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    results.append(f"{title}: {result}")
                else:
                    results.append(f"{title}: Failed - empty content")
                    
            except Exception as e:
                results.append(f"{title}: Failed - {str(e)}")
        
        return "\n".join(results)

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
            return " Please provide a request. You can add books by saying 'add [book title]' or 'download samples'."
        
        prompt = prompt.strip().lower()
        
        try:
            # Handle sample book downloads
            if any(keyword in prompt for keyword in ["download samples", "sample books", "get samples"]):
                return self.download_sample_books()
            
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
                
                return " Please specify a book title. Example: 'add Pride and Prejudice' or try 'download samples' for quick setup."
            
            elif "list" in prompt or "show" in prompt or "available" in prompt:
                books = self.get_available_books()
                if books:
                    return f" Available books:\n" + "\n".join(f" {book}" for book in books)
                else:
                    return " No books have been added to the system yet. Try 'download samples' to get started!"
            
            else:
                return " I can help you add books to the system. Try: 'add [book title]', 'list books', or 'download samples' for quick setup."
                
        except Exception as e:
            return f" An unexpected error occurred: {str(e)}"

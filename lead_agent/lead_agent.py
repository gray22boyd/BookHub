import os
import requests
from typing import Optional
from config.settings import config
from agents.answer_agent import AnswerAgent
from agents.organizer_agent import OrganizingAgent

class LeadAgent:
    """
    Main routing agent that directs user queries to appropriate specialized agents.
    Simplified to handle only book Q&A and book organization tasks.
    """
    
    def __init__(self):
        """Initialize the lead agent with organizer agent."""
        try:
            self.organizer_agent = OrganizingAgent()
        except Exception as e:
            print(f" Warning: OrganizingAgent initialization failed: {e}")
            self.organizer_agent = None
        

        
        # Ensure books directory exists
        self.books_dir = "books"
        os.makedirs(self.books_dir, exist_ok=True)

    def download_gutenberg_book(self, title: str) -> str:
        """
        Download a book directly from Project Gutenberg by searching for it dynamically.
        
        Args:
            title: Book title to download
            
        Returns:
            Success or error message
        """
        try:
            # Clean title for filename
            clean_title = title.lower().strip()
            filename = clean_title.replace(" ", "_").replace("'", "").replace(",", "").replace(".", "") + ".txt"
            filepath = os.path.join(self.books_dir, filename)
            
            # Check if already exists
            if os.path.exists(filepath):
                return f"📚 '{title.title()}' already downloaded to {filepath}"
            
            print(f"Searching for '{title}' on Project Gutenberg...")
            
            # Step 1: Search for the book using Gutenberg API
            search_url = f"https://gutendx.com/books/?search={title}"
            
            # Add proper headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Use session with proper SSL settings
            session = requests.Session()
            session.headers.update(headers)
            
            search_response = session.get(search_url, timeout=30, verify=True)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if not search_data.get("results"):
                return f"❌ No books found for '{title}' on Project Gutenberg. Try checking the spelling or using a different title format."
            
            # Get the first result (most relevant)
            book = search_data["results"][0]
            book_id = book["id"]
            book_title = book.get("title", title)
            
            print(f"Found '{book_title}' (ID: {book_id}) - downloading...")
            
            # Step 2: Try multiple download URLs in order of preference
            download_urls = [
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
            ]
            
            download_success = False
            for url in download_urls:
                try:
                    print(f"Trying download URL: {url}")
                    response = session.get(url, timeout=30, verify=True)
                    response.raise_for_status()
                    
                    if response.text and len(response.text) > 1000:
                        # Save to file
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                        download_success = True
                        break
                    else:
                        print(f"Content too short from {url}, trying next...")
                        
                except requests.RequestException as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
            
            if not download_success:
                return f"❌ Failed to download '{title}'. The book may not be available in plain text format."
            
            print(f"Successfully saved to {filepath}")
            
            # Step 3: Automatically add to BookAgent system
            if self.organizer_agent:
                try:
                    add_result = self.organizer_agent.add_book_from_file(book_title, filepath)
                    if "Successfully added" in add_result:
                        return f"✅ '{book_title}' downloaded and added to BookAgent successfully!\n📁 Saved to: {filepath}\n🔍 You can now ask questions about this book."
                    else:
                        return f"✅ '{book_title}' downloaded successfully to {filepath}\n⚠️ Auto-add to BookAgent failed: {add_result}\nYou can manually add it later."
                except Exception as e:
                    return f"✅ '{book_title}' downloaded successfully to {filepath}\n⚠️ Auto-add to BookAgent failed: {str(e)}\nYou can manually add it later."
            else:
                return f"✅ '{book_title}' downloaded successfully to {filepath}\n💡 Use the organizer to add it to BookAgent."
                
        except requests.exceptions.Timeout as e:
            return f"❌ Request timed out for '{title}'. The server may be slow or overloaded. Try again in a few minutes."
        except requests.exceptions.ConnectionError as e:
            return f"❌ Connection failed for '{title}': {str(e)}. Check your internet connection."
        except requests.exceptions.HTTPError as e:
            return f"❌ HTTP error for '{title}': {e.response.status_code} - {str(e)}"
        except requests.RequestException as e:
            return f"❌ Failed to search/download '{title}': Network error - {str(e)}"
        except Exception as e:
            return f"❌ Error downloading '{title}': {str(e)}"

    def classify_prompt(self, prompt: str) -> str:
        """
        Classify user prompts into categories:
        - 'download': For downloading books from Gutenberg
        - 'organize': For adding/uploading/ingesting books
        - 'answer': For all book-related questions (default)
        """
        prompt_lower = prompt.lower().strip()
        
        if prompt_lower.startswith("download "):
            return "download"
        elif any(keyword in prompt_lower for keyword in ["add", "upload", "ingest", "download samples", "sample books"]):
            return "organize"
        else:
            return "answer"  # Default to book Q&A for everything else

    def route(self, prompt: str, book_title: Optional[str] = None) -> str:
        """
        Route user prompts to appropriate agents based on classification.
        
        Args:
            prompt: User's input query
            book_title: Selected book title (required for Q&A)
            
        Returns:
            Response from the appropriate agent
        """
        classification = self.classify_prompt(prompt)
        
        try:
            if classification == "download":
                # Extract book title from download command
                if prompt.lower().startswith("download "):
                    book_title_to_download = prompt[9:].strip()  # Remove "download " prefix
                    if not book_title_to_download:
                        return "❌ Please specify a book title. Example: 'download Pride and Prejudice'"
                    return self.download_gutenberg_book(book_title_to_download)
                else:
                    return "❌ Please use format: 'download [book title]'"
            
            elif classification == "organize":
                if self.organizer_agent is None:
                    return " Book organization service is not available. Please check your configuration."
                return self.organizer_agent.handle(prompt)
                
            elif classification == "answer":
                if not book_title:
                    return " Please select a book first to ask questions about it."
                    
                try:
                    answer_agent = AnswerAgent(book_title)
                    return answer_agent.answer(prompt)
                except Exception as e:
                    return f" Error accessing book '{book_title}': {str(e)}. Make sure the book has been added to the system."
                    
            else:
                return " I'm not sure how to process that request. Try asking about a book, adding a new book, or downloading with 'download [title]'."
                
        except Exception as e:
            return f" An unexpected error occurred: {str(e)}"
    
    def handle_prompt(self, prompt: str, book_title: Optional[str] = None) -> str:
        """
        Alias for route method to maintain compatibility.
        """
        return self.route(prompt, book_title)



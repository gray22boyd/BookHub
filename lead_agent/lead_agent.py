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

    def classify_prompt(self, prompt: str) -> str:
        """
        Classify user prompts into two categories:
        - 'organize': For adding/uploading/ingesting books
        - 'answer': For all book-related questions (default)
        """
        if any(keyword in prompt.lower() for keyword in ["add", "upload", "ingest"]):
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
            if classification == "organize":
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
                return " I'm not sure how to process that request. Try asking about a book or adding a new book."
                
        except Exception as e:
            return f" An unexpected error occurred: {str(e)}"
    
    def handle_prompt(self, prompt: str, book_title: Optional[str] = None) -> str:
        """
        Alias for route method to maintain compatibility.
        """
        return self.route(prompt, book_title)

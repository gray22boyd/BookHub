from typing import List, Optional
from config.settings import config
from enum import Enum
from exceptions import ApiConnectionError
from Endpoints.Gutenberg import Gutenberg

class BookSource(Enum):
    PROJECT_GUTENBERG = "gutenberg"
    OPENLIBRARY = "openlibrary"
    GOOGLE_BOOKS = "google books"
    AMAZON_BOOKS = "amazon books"
    STORAGE = "storage"

class BookApiClient:
    def __init__(self, source: Enum = BookSource.STORAGE, api_key: str = None):
        self.source = source
        self.api_key = api_key

    def get_book_text(self, book_title: str, author_name: str, translator: str = None):
        """
        Get a book from the API.

        Args:
            book_title: Title of the book to search for

        Returns:
            Full text file of the book requested
        """
        if self.source == BookSource.PROJECT_GUTENBERG:
            return Gutenberg(book_title, author_name).get_book_text()
        else:
            raise ValueError(f"Unsupported book source: {self.source}")
    
    def header_and_footer_removal(self, book_text: str) -> str:
        pass


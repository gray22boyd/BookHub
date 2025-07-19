from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Endpoints.Gutenberg import Gutenberg
from exceptions import ApiConnectionError
import nltk
nltk.download('punkt')

class BookSource(Enum):
    PROJECT_GUTENBERG = "gutenberg"
    OPENLIBRARY = "openlibrary"
    GOOGLE_BOOKS = "google books"
    AMAZON_BOOKS = "amazon books"
    STORAGE = "storage"

class BookApiClient:
    def __init__(self, book_title: str, source: Enum = BookSource.STORAGE,  api_key: str | None = None, author_name: str | None = None, translator: str | None = None):
        self.source = source
        self.api_key = api_key
        self.book_title = book_title
        self.author_name = author_name
        self.translator = translator
        self.endpoint = self.initalize_endpoint()

    def initalize_endpoint(self):
        if self.source == BookSource.PROJECT_GUTENBERG:
            if self.book_title is None:
                raise ValueError("book_title is required for Gutenberg source")
            return Gutenberg(book_title=self.book_title, author_name=self.author_name, translator=self.translator)
        else:
            raise ValueError(f"Unsupported book source: {self.source}")

    def tokenize_text(self):
        from nltk.tokenize import sent_tokenize
        text = self.endpoint.remove_gutenberg_header_footer()

        sentences = sent_tokenize(text)
        return sentences

    
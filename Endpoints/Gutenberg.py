import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import ApiConnectionError, BookNotFoundException

class Gutenberg:
    def __init__(self, book_title: str, author_name: str = None, translator: str = None):
        self.book_title = book_title
        self.author_name = author_name
        self.translator = translator
    
    def get_book_text(self):
        """
        Get the text of the book.
        """
        url = "https://gutendex.com/books/?search=Anna Karenina"
        print(f"Endpoint URL: {url}")
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise ApiConnectionError(f"Failed to connect to Gutenberg API: {response.status_code}")
        data = response.json()
        text_url = data["results"][0]["formats"]["text/plain; charset=us-ascii"]
        response = requests.get(text_url, timeout=30)
        if response.status_code != 200:
            raise ApiConnectionError(f"Failed to connect to Gutenberg API: {response.status_code}")
        return response.text

        #TODO: Implement the logic to check for the translator and author name

    def remove_gutenberg_header_footer(self,book_text: str) -> str:
    # Find the line that marks the end of the header
        start_index = 0
        # Removing the header
        for i, line in enumerate(book_text):
            print(f"Line {i}: {line}")
            if "*** START OF THE PROJECT GUTENBERG EBOOK" in line.upper():
                start_index = i + 1  # Skip this line too
                break
        

    
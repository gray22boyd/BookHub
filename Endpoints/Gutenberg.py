import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exceptions import ApiConnectionError, BookNotFoundException

class Gutenberg:
    def __init__(self, book_title: str, author_name: str | None = None, translator: str | None = None):
        self.book_title = book_title
        self.author_name = author_name
        self.translator = translator
    
    def get_book_text(self):
        """
        Get the text of the book.
        """
        url = f"https://gutendex.com/books/?search={self.book_title}"
        print(f"Endpoint URL: {url}")
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise ApiConnectionError(f"Failed to connect to Gutenberg API: {response.status_code}")
        data = response.json()
        text_url = data["results"][0]["formats"]["text/plain"]
        response = requests.get(text_url, timeout=30)
        if response.status_code != 200:
            raise ApiConnectionError(f"Failed to connect to Gutenberg API: {response.status_code}")
        return response.text
        #TODO: Implement logic to grab textfile no matter the keywording
        #TODO: Implement the logic to check for the translator and author name

    def remove_gutenberg_header_footer(self) -> str:
        # Split the text into lines for processing
        lines = self.get_book_text().split('\n')
        
        # Find the line that marks the end of the header
        start_index = 0
        # Removing the header
        for i, line in enumerate(lines):
            if "*** START OF THE PROJECT GUTENBERG EBOOK" in line.upper():
                start_index = i + 1
                break
        
        # Removing the footer
        end_index = len(lines)
        for i, line in enumerate(lines):
            if "*** END OF THE PROJECT GUTENBERG EBOOK" in line.upper():
                end_index = i
                break   
        
        # Join the filtered lines back into a string
        return '\n'.join(lines[start_index:end_index])

    
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.BookApiClient import BookApiClient, BookSource

def __main__():
    book_api_client = BookApiClient(source=BookSource.PROJECT_GUTENBERG)
    book_text = book_api_client.get_book_text(book_title="Anna Karenina", author_name="Tolstoy, Leo, graf")
    print(book_text)

if __name__ == "__main__":
    __main__()


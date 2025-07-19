import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Endpoints.Gutenberg import Gutenberg as gutenberg
from services.BookApiClient import BookApiClient, BookSource


def __main__():
    book_inst = BookApiClient(source=BookSource.PROJECT_GUTENBERG, book_title="Queen of Spades")
    sentences = book_inst.tokenize_text()
    print(sentences)

if __name__ == "__main__":
    (__main__())  


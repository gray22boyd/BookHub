import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Endpoints.Gutenberg import Gutenberg as gutenberg


def __main__():
    with open("C:/Users/LukeHannon/Desktop/Python Projects/BookHub/Queen of Spades.txt", "r", encoding="utf-8") as file:
        book_text = file.readlines()
    book_inst = gutenberg("Queen of Spades")
    cleaned_text = book_inst.remove_gutenberg_header_footer(book_text)
    print(book_text)


if __name__ == "__main__":
    (__main__())  


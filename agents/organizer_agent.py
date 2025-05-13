from ftplib import print_line

import requests
import textwrap
import re

# Calling Gutenberg API to locate book text file

# Def bookNameFormatter - Formats the title of the book into the needed String to call the API

def locate_book(book_title):
    URL = f"https://gutendex.com/books/?search={book_title}"
    headers = {
        "Content-Type": "application/json"
    }
    resp = requests.get(URL, headers=headers)
    data = resp.json()
    text_file_url = data["results"][0]["formats"]["text/plain; charset=us-ascii"]
    text_json = requests.get(text_file_url)
    text = text_json.text
    return text

def chunk_book(text):
    # Due to how text files are formatted as strings we must first rid the text of carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Parses by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)




my_text = locate_book("Anna Karenina")
chunk_book(my_text)





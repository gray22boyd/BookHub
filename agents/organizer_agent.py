from os import environ

import requests
import textwrap
import re
from typing import List
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetirevalQA

#Open_AI_Key = env.OPENAI_KEy

#Calling Gutenberg API to locate book text file

#Def bookNameFormatter - Formats the title of the book into the needed String to call the API

class OrganizingAgent:

    def locate_book(book_title):
        URL = f"https://gutendex.com/books/?search={book_title}"
        headers = {
            "Content-Type": "application/json"
        }
        resp = requests.get(URL, headers=headers)
        data = resp.json()
        text_file_url = data["results"][0]["formats"]["text/plain; charset=us-ascii"]
        # Use UTF link to pull the text from the web
        text_json = requests.get(text_file_url)
        text = text_json.text
        return text

    def chunk_book(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base") # For text_embedding-ada-002 or GPT-4

        # Due to how text files are formatted as strings we must first rid the text of carriage returns
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Parses by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_token_count = 0

        for para in paragraphs:
            tokens = tokenizer.encode(para)
            if current_token_count + len(tokens) > max_tokens:
                # finalize the current chunk
                full_chunk = " ".join(current_chunk)
                chunks.append(full_chunk)

                # start new chunk with overlap
                if overlap > 0:
                    overlap_tokens = tokens[-overlap:]
                    overlap_text = tokenizer.decode(overlap_tokens)
                    current_chunk = [overlap_text]
                    current_token_count = len(overlap_tokens)
                else:
                    current_chunk = []
                    current_token_count = 0

            current_chunk.append(para)
            current_token_count += len(tokens)

            # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks










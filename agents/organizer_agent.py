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
from langchain.chains import RetrievalQA

# Example: Open_AI_Key = environ["OPENAI_KEY"]

# OrganizingAgent: Tools for working with book text from Project Gutenberg

class OrganizingAgent:
    @staticmethod
    def locate_book(book_title):
        """
        Finds and downloads a plain text version of a book from the Gutendex API.
        """
        URL = f"https://gutendex.com/books/?search={book_title}"
        headers = {
            "Content-Type": "application/json"
        }
        resp = requests.get(URL, headers=headers)
        data = resp.json()
        text_file_url = data["results"][0]["formats"]["text/plain; charset=us-ascii"]
        # Download the book text
        text_json = requests.get(text_file_url)
        text = text_json.text
        return text

    @staticmethod
    def chunk_book(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
        """
        Chunks a book into overlapping segments suitable for embeddings/vector databases.
        """
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")  # For text_embedding-ada-002 or GPT-4

        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Split by paragraphs
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
                if overlap > 0 and len(tokens) > 0:
                    overlap_tokens = tokens[-overlap:]
                    overlap_text = tokenizer.decode(overlap_tokens)
                    current_chunk = [overlap_text]
                    current_token_count = len(overlap_tokens)
                else:
                    current_chunk = []
                    current_token_count = 0

            current_chunk.append(para)
            current_token_count += len(tokens)

        # Add the final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

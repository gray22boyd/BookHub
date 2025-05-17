from dotenv import load_dotenv
import os

import requests
import textwrap
import re
from typing import List
from time import sleep
import tiktoken
import time
import faiss
import numpy as np
from dataclasses import dataclass
import json
# Example: Open_AI_Key = environ["OPENAI_KEY"]

# OrganizingAgent: Tools for working with book text from Project Gutenberg

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class BookRecord:
    title: str
    first_chunk: int
    last_chunk: int

import openai
openai.api_key = OPENAI_API_KEY

class OrganizingAgent:
    def __init__(self):
        self.chunk_texts = []
        self.index = faiss.IndexFlatL2(1536)
        self.meta_data: List[BookRecord] = []

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
    def chunk_text_tiktoken(text, max_tokens=800, overlap=100):
        """
        Splits `text` into overlapping chunks, each with up to `max_tokens` tokens.
        Uses OpenAI's tiktoken tokenizer.

        Args:
            text (str): The full text to chunk.
            max_tokens (int): Max tokens per chunk (800 is safe for most embeddings).
            overlap (int): Overlap tokens between chunks (for context continuity).

        Returns:
            list of str: List of text chunks.
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")  # Best for OpenAI embeddings

        # Normalize newlines for consistent splitting
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Split into paragraphs for semantic boundaries
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_token_count = 0

        for para in paragraphs:
            tokens = tokenizer.encode(para)
            if current_token_count + len(tokens) > max_tokens:
                # Finalize and store the current chunk
                full_chunk = " ".join(current_chunk)
                chunks.append(full_chunk)

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 0:
                    # Take the last `overlap` tokens from the previous chunk
                    previous_text = " ".join(current_chunk)
                    prev_tokens = tokenizer.encode(previous_text)
                    overlap_tokens = prev_tokens[-overlap:]
                    overlap_text = tokenizer.decode(overlap_tokens)
                    current_chunk = [overlap_text]
                    current_token_count = len(overlap_tokens)
                else:
                    current_chunk = []
                    current_token_count = 0

            current_chunk.append(para)
            current_token_count += len(tokens)

        # Add any leftover chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    # Passing functions into parameter, so that we may try different models or implementations
    # for embedding or storing later on
    def embed_and_store_chunks(chunks, embed_fn, store_fn, batch_size=10, delay=2):
        """
        Embeds and stores text chunks in batches.

        Args:
            chunks (list of str): The text chunks to embed.
            embed_fn (callable): Function that takes a list of strings and returns list of embeddings.
            store_fn (callable): Function that stores a batch of (chunk, embedding) pairs.
            batch_size (int): Number of chunks per batch.
            delay (int): Seconds to wait between batches.
        """
        batch = []
        indices = []
        for i, chunk in enumerate(chunks):
            batch.append(chunk)
            indices.append(i)
            if len(batch) == batch_size or i == len(chunks) - 1:
                # 1. Embed the batch
                embeddings = embed_fn(batch)  # Should return a list of vectors
                # 2. Store the embeddings, keeping mapping to chunk indices if needed
                store_fn(batch, embeddings, indices)
                # 3. Wait to avoid API rate limits
                time.sleep(delay)
                batch = []
                indices = []
    @staticmethod
    def embed_fn(chunks):
        import openai
        response = openai.embeddings.create(
            input=chunks,
            model="text-embedding-ada-002"
        )
        return [r.embedding for r in response.data]

    def add_book_metadata(self, title: str, start_idx: int, end_idx: int):
        self.meta_data.append(BookRecord(title, first_chunk=start_idx, last_chunk=end_idx))

    def store_fn(self, batch, embeddings, indices):
        arr = np.array(embeddings).astype('float32')
        self.index.add(arr)
        self.chunk_texts.extend(batch)

    def save_to_disk(self, index_path="my_index.faiss", chunks_path="chunk_texts.json", meta_path="meta_data.json"):
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_texts, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in self.meta_data], f)

    def load_from_disk(self, index_path="my_index.faiss", chunks_path="chunk_texts.json", meta_path="meta_data.json"):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunk_texts = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            self.meta_data = [BookRecord(**r) for r in records]

    def process_data(self, book_title, ):
        # Actual 1. Load current data and check if the book is already stored in our data base
            # If so pull the relevant chunks
            # else continue with process

        # Have agent correctly format the book title?


        # 1. Download book
        print("Downloading book...")
        book_text = self.locate_book(book_title)

        # 2. Chunk it
        print("Chunking text...")
        chunks = self.chunk_text_tiktoken(book_text)

        # Check what our first index fo the new book will be d
        start_index = self.index.ntotal

        # 3. Embed and store
        print("Embedding and storing chunks...")
        print("Chunks to embed:", len(chunks))
        self.embed_and_store_chunks(
            chunks,
            embed_fn=self.embed_fn,
            store_fn=self.store_fn
        )

        # 4. Confirm FAISS storage
        print("Total vectors stored in FAISS:", self.index.ntotal)

        end_index = self.index.ntotal - 1

        self.add_book_metadata(book_title, start_index, end_index)

        # 5. View sample chunk and metadata
        print("Sample chunk:", self.chunk_texts[20][:300], "...")




import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Embedding settings
    EMBEDDING_MODEL: str = "intfloat/e5-large-v2"
    EMBEDDING_DIMENSIONS: int = 1024
    
    # Text processing (optimized for sentence-aware chunking)
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100  # Increased overlap for better context continuity
    
    # Retrieval settings (optimized for hybrid retrieval)
    RETRIEVAL_K: int = 8  # Increased to get more candidates for reranking
    
    # Paths
    FAISS_INDEXES_DIR: str = "faiss_indexes"
    
    # OpenAI settings (optimized for better responses)
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.1  # Lower temperature for more consistent answers
    OPENAI_MAX_TOKENS: int = 750  # Increased for more detailed responses

# Global config instance
config = Config() 
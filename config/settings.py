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
    
    # Text processing
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    
    # Retrieval settings
    RETRIEVAL_K: int = 5
    
    # Paths
    FAISS_INDEXES_DIR: str = "faiss_indexes"
    
    # OpenAI settings
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.2
    OPENAI_MAX_TOKENS: int = 500

# Global config instance
config = Config() 
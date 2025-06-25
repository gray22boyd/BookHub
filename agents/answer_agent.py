import os
import re
from typing import Optional
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from config.settings import config
from services.embedding_service import EmbeddingService

class AnswerAgent:
    def __init__(self, book_directory: str):
        if not book_directory or not book_directory.strip():
            raise ValueError("Book directory cannot be empty")
            
        self.book_directory = book_directory.strip()
        # Use the directory name directly instead of sanitizing
        self.sanitized_title = self.book_directory
        
        self.embedding_service = EmbeddingService()
        embeddings = self.embedding_service.get_embeddings()
        
        index_path = os.path.join(config.FAISS_INDEXES_DIR, self.sanitized_title)
        
        if not os.path.exists(index_path):
            available_books = self._get_available_books()
            raise FileNotFoundError(
                f"No index found for book directory '{book_directory}'. "
                f"Available books: {', '.join(available_books) if available_books else 'None'}"
            )
        
        try:
            self.vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(
                    temperature=config.OPENAI_TEMPERATURE,
                    max_tokens=config.OPENAI_MAX_TOKENS
                ),
                chain_type="stuff",
                retriever=retriever
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QA system for '{book_directory}': {str(e)}")
    

    
    def _get_available_books(self) -> list:
        try:
            if not os.path.exists(config.FAISS_INDEXES_DIR):
                return []
            return [
                name for name in os.listdir(config.FAISS_INDEXES_DIR)
                if os.path.isdir(os.path.join(config.FAISS_INDEXES_DIR, name))
            ]
        except Exception:
            return []
    
    def _detect_specific_queries(self, query: str) -> Optional[str]:
        sentence_match = re.search(r'what is the (\d+)(?:st|nd|rd|th)? sentence', query.lower())
        if sentence_match:
            sentence_num = int(sentence_match.group(1))
            return self._get_sentence_by_number(sentence_num)
        
        paragraph_match = re.search(r'what is the (\d+)(?:st|nd|rd|th)? paragraph', query.lower())
        if paragraph_match:
            paragraph_num = int(paragraph_match.group(1))
            return self._get_paragraph_by_number(paragraph_num)
        
        return None
    
    def _get_sentence_by_number(self, sentence_num: int) -> str:
        try:
            all_docs = self.vectorstore.similarity_search("", k=100)
            full_text = " ".join([doc.page_content for doc in all_docs])
            full_text = re.sub(r'^passage:\s*', '', full_text, flags=re.MULTILINE)
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentence_num < 1 or sentence_num > len(sentences):
                return f"Sentence number {sentence_num} is out of range. The book appears to have approximately {len(sentences)} sentences."
            
            return f"Sentence {sentence_num}: {sentences[sentence_num - 1]}"
            
        except Exception as e:
            return f"Sorry, I couldn't retrieve sentence {sentence_num}: {str(e)}"
    
    def _get_paragraph_by_number(self, paragraph_num: int) -> str:
        try:
            all_docs = self.vectorstore.similarity_search("", k=100)
            full_text = " ".join([doc.page_content for doc in all_docs])
            full_text = re.sub(r'^passage:\s*', '', full_text, flags=re.MULTILINE)
            paragraphs = re.split(r'\n\s*\n', full_text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if paragraph_num < 1 or paragraph_num > len(paragraphs):
                return f"Paragraph number {paragraph_num} is out of range. The book appears to have approximately {len(paragraphs)} paragraphs."
            
            return f"Paragraph {paragraph_num}: {paragraphs[paragraph_num - 1]}"
            
        except Exception as e:
            return f"Sorry, I couldn't retrieve paragraph {paragraph_num}: {str(e)}"
    
    def answer(self, query: str) -> str:
        if not query or not query.strip():
            return "Please provide a question about the book."
        
        try:
            specific_response = self._detect_specific_queries(query)
            if specific_response:
                return specific_response
            
            prefixed_query = f"query: {query.strip()}"
            result = self.qa_chain.run(prefixed_query)
            
            if not result or result.strip() == "":
                # Convert directory name back to readable format for user-facing messages
                book_display = self.book_directory.replace('_', ' ').title()
                return f"I couldn't find relevant information in '{book_display}' to answer your question. Try rephrasing or asking about a different aspect of the book."
            
            return result
            
        except Exception as e:
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

import os
import re
import joblib
from typing import Optional, List, Dict, Any
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
            # Load the vector store
            self.vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Initialize hybrid retrieval
            self._setup_hybrid_retrieval()
            
            # Initialize cross-encoder for reranking
            self._setup_reranker()
            
            # Setup cache
            self.cache_dir = os.path.join(index_path, "cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Initialize QA chain with better prompt
            self.llm = OpenAI(
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            
            print(f"Successfully initialized AnswerAgent for '{book_directory}'")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QA system for '{book_directory}': {str(e)}")
    
    def _setup_hybrid_retrieval(self):
        """Setup hybrid retrieval combining semantic search and BM25."""
        try:
            # Get all documents from vectorstore for BM25
            all_docs = self.vectorstore.similarity_search("", k=1000)  # Get all docs
            texts = [doc.page_content for doc in all_docs]
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_texts(texts, k=config.RETRIEVAL_K)
            
            # Create semantic retriever
            self.semantic_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            )
            
            # Create ensemble retriever (70% semantic, 30% BM25)
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.bm25_retriever], 
                weights=[0.7, 0.3]
            )
            
            print(f"Setup hybrid retrieval with {len(texts)} documents")
            
        except Exception as e:
            print(f"Warning: Failed to setup hybrid retrieval, falling back to semantic only: {e}")
            self.hybrid_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            )
    
    def _setup_reranker(self):
        """Setup cross-encoder for reranking retrieved documents."""
        try:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("Loaded cross-encoder reranking model")
        except Exception as e:
            print(f"Warning: Failed to load reranking model: {e}")
            self.rerank_tokenizer = None
            self.rerank_model = None
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not self.rerank_model or not docs:
            return docs
        
        try:
            # Prepare inputs for cross-encoder
            doc_texts = [doc.page_content.replace("passage: ", "") for doc in docs]
            inputs = [f"{query} [SEP] {doc_text}" for doc_text in doc_texts]
            
            # Tokenize and get scores
            encoded = self.rerank_tokenizer(
                inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.rerank_model(**encoded)
                scores = outputs.logits.squeeze().tolist()
            
            # Handle single document case
            if not isinstance(scores, list):
                scores = [scores]
            
            # Sort documents by score (highest first)
            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            reranked_docs = [doc for _, doc in scored_docs]
            
            print(f"Reranked {len(docs)} documents. Top score: {max(scores):.3f}")
            return reranked_docs
            
        except Exception as e:
            print(f"Warning: Reranking failed, using original order: {e}")
            return docs
    
    def _get_cached_result(self, query: str) -> Optional[str]:
        """Get cached result for query if available."""
        try:
            cache_file = os.path.join(self.cache_dir, f"query_{hash(query) % 10000}.pkl")
            if os.path.exists(cache_file):
                return joblib.load(cache_file)
        except Exception:
            pass
        return None
    
    def _cache_result(self, query: str, result: str):
        """Cache query result."""
        try:
            cache_file = os.path.join(self.cache_dir, f"query_{hash(query) % 10000}.pkl")
            joblib.dump(result, cache_file)
        except Exception:
            pass
    
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
        """Get specific sentence from full book text."""
        try:
            # Load full book text
            text_path = os.path.join(config.FAISS_INDEXES_DIR, self.sanitized_title, "book_text.txt")
            if not os.path.exists(text_path):
                return f"Full book text not found. The book may need to be re-indexed."
            
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if sentence_num < 1 or sentence_num > len(sentences):
                return f"Sentence number {sentence_num} is out of range. The book has {len(sentences)} sentences."

            return f"Sentence {sentence_num}: {sentences[sentence_num - 1]}"
            
        except Exception as e:
            return f"Failed to fetch sentence {sentence_num}: {e}"
    
    def _get_paragraph_by_number(self, paragraph_num: int) -> str:
        """Get specific paragraph from full book text."""
        try:
            # Load full book text
            text_path = os.path.join(config.FAISS_INDEXES_DIR, self.sanitized_title, "book_text.txt")
            if not os.path.exists(text_path):
                return f"Full book text not found. The book may need to be re-indexed."
            
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            if paragraph_num < 1 or paragraph_num > len(paragraphs):
                return f"Paragraph number {paragraph_num} is out of range. The book has {len(paragraphs)} paragraphs."

            return f"Paragraph {paragraph_num}: {paragraphs[paragraph_num - 1]}"
            
        except Exception as e:
            return f"Failed to fetch paragraph {paragraph_num}: {e}"
    
    def _verify_answer_quality(self, answer: str) -> bool:
        """Verify if the answer meets quality standards."""
        if not answer or len(answer.strip()) < 10:
            return False
        
        low_quality_indicators = [
            "i don't know",
            "i cannot answer",
            "no information",
            "not mentioned",
            "unclear",
            "insufficient information"
        ]
        
        answer_lower = answer.lower()
        return not any(indicator in answer_lower for indicator in low_quality_indicators)
    
    def _format_retrieval_prompt(self, query: str, retrieved_docs: List[Document]) -> str:
        """Format the prompt with retrieved context in a structured way."""
        # Clean up passage prefixes
        contexts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content.replace("passage: ", "").strip()
            contexts.append(f"{i}. {content}")
        
        context_text = "\n".join(contexts)
        
        prompt = f"""Based on the following passages from the book, answer the question accurately and comprehensively.

Context:
{context_text}

Question: {query}

Instructions:
- Use only the information provided in the context above
- If the context doesn't contain enough information, say so
- Be specific and cite relevant details from the passages
- Keep your answer focused and relevant to the question

Answer:"""
        
        return prompt
    
    def answer(self, query: str) -> str:
        if not query or not query.strip():
            return "Please provide a question about the book."
        
        query = query.strip()
        
        # Check cache first
        cached_result = self._get_cached_result(query)
        if cached_result:
            print("Retrieved answer from cache")
            return cached_result
        
        try:
            # Handle specific sentence/paragraph queries
            specific_response = self._detect_specific_queries(query)
            if specific_response:
                self._cache_result(query, specific_response)
                return specific_response
            
            # Step 1: Hybrid retrieval
            print(f"Performing hybrid retrieval for: {query}")
            prefixed_query = f"query: {query}"
            retrieved_docs = self.hybrid_retriever.get_relevant_documents(prefixed_query)
            
            if not retrieved_docs:
                no_docs_msg = "I couldn't find relevant information to answer your question. Try rephrasing or asking about a different aspect of the book."
                self._cache_result(query, no_docs_msg)
                return no_docs_msg
            
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Rerank documents
            reranked_docs = self._rerank_documents(query, retrieved_docs)
            
            # Step 3: Format structured prompt
            formatted_prompt = self._format_retrieval_prompt(query, reranked_docs[:5])  # Use top 5
            
            # Debug output
            print(f"Top retrieved chunk: {reranked_docs[0].page_content[:100]}...")
            
            # Step 4: Generate answer
            result = self.llm(formatted_prompt)
            
            # Step 5: Verify answer quality
            if not self._verify_answer_quality(result):
                fallback_msg = "I couldn't find a clear answer to your question in the book. Try rephrasing your question or asking about a different topic."
                self._cache_result(query, fallback_msg)
                return fallback_msg
            
            # Cache and return result
            self._cache_result(query, result)
            return result
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
            print(f"Error in answer generation: {e}")
            return error_msg

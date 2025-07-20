import os
import re
import joblib
import json
from typing import Optional, List, Dict, Any
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.settings import config
from services.embedding_service import EmbeddingService

class AnswerAgent:
    """Handles question answering for books using hybrid retrieval and LLM generation."""
    
    def __init__(self, book_directory: str):
        """Initialize the answer agent with a specific book directory."""
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
            
            # Load sentence index for positional lookups
            sentence_index_path = os.path.join(index_path, "sentence_index.json")
            self.sentence_index = None
            if os.path.exists(sentence_index_path):
                try:
                    with open(sentence_index_path, "r", encoding="utf-8") as f:
                        self.sentence_index = json.load(f)
                    print(f"📚 Loaded sentence index with {len(self.sentence_index)} sentences")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ Warning: Could not load sentence index: {e}")
                    self.sentence_index = None
            else:
                print(f"⚠️ Warning: No sentence index found at {sentence_index_path}")
            
            # Initialize hybrid retrieval
            self._setup_hybrid_retrieval()
            
            # Initialize cross-encoder for reranking
            self._setup_reranker()
            
            # Setup cache
            self.cache_dir = os.path.join(index_path, "cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Initialize QA chain with better prompt
            self.llm = ChatOpenAI(
                model="gpt-4-turbo",
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
        """Get list of available books from the indexes directory."""
        try:
            if not os.path.exists(config.FAISS_INDEXES_DIR):
                return []
            return [
                name for name in os.listdir(config.FAISS_INDEXES_DIR)
                if os.path.isdir(os.path.join(config.FAISS_INDEXES_DIR, name))
            ]
        except Exception:
            return []
    
    def _get_sentence_by_position(self, sentence_num: int) -> Optional[str]:
        """Get specific sentence using pre-loaded sentence index."""
        if self.sentence_index is None:
            return None
        
        if sentence_num < 1:
            return f"❌ Invalid sentence number: {sentence_num}. Sentence numbers start from 1."
        
        if sentence_num > len(self.sentence_index):
            return f"The book only has {len(self.sentence_index)} sentences."
        
        # Get the sentence (index is 1-based)
        sentence_data = self.sentence_index[sentence_num - 1]
        return f"📝 Sentence {sentence_num}: {sentence_data['text']}"

    def _detect_positional_sentence_query(self, query: str) -> Optional[str]:
        """Detect and handle positional sentence queries using pre-loaded index."""
        import re
        
        # Enhanced sentence patterns
        patterns = [
            r'what is the (\d+)(?:st|nd|rd|th)? sentence',
            r'show me (?:the )?(\d+)(?:st|nd|rd|th)? sentence',
            r'give me (?:the )?(\d+)(?:st|nd|rd|th)? sentence',
            r'get (?:the )?(\d+)(?:st|nd|rd|th)? sentence',
            r'find (?:the )?(\d+)(?:st|nd|rd|th)? sentence',
            r'(?:the )?(\d+)(?:st|nd|rd|th)? sentence',
            r'sentence (\d+)',
            r'what is sentence (\d+)',
            r'show sentence (\d+)',
            r'what is the (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth) sentence'
        ]
        
        # Number word to digit mapping
        word_to_num = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
        }
        
        sentence_num = None
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                matched_value = match.group(1)
                if matched_value.isdigit():
                    sentence_num = int(matched_value)
                elif matched_value in word_to_num:
                    sentence_num = word_to_num[matched_value]
                break
        
        if sentence_num is None:
            return None
        
        print(f"🎯 DEBUG: Positional sentence lookup for sentence #{sentence_num}")
        return self._get_sentence_by_position(sentence_num)



    
    
    def _get_paragraph_by_number(self, paragraph_num: int) -> str:
        """Get specific paragraph from structured paragraph index."""
        try:
            # Load paragraph index
            paragraph_index_path = os.path.join(config.FAISS_INDEXES_DIR, self.sanitized_title, "paragraph_index.json")
            if not os.path.exists(paragraph_index_path):
                return f"📚 Paragraph index not found. The book may need to be re-indexed with the latest version."
            
            with open(paragraph_index_path, "r", encoding="utf-8") as f:
                paragraph_index = json.load(f)
            
            # Validate paragraph number
            if paragraph_num < 1:
                return f"❌ Invalid paragraph number: {paragraph_num}. Paragraph numbers start from 1."
            
            if paragraph_num > len(paragraph_index):
                return f"❌ Paragraph number {paragraph_num} is out of range. This book has {len(paragraph_index)} paragraphs available."
            
            # Get the paragraph (index is 1-based)
            paragraph_data = paragraph_index[paragraph_num - 1]
            return f"📄 Paragraph {paragraph_num}: {paragraph_data['text']}"
            
        except json.JSONDecodeError as e:
            return f"❌ Error reading paragraph index: {e}"
        except Exception as e:
            return f"❌ Failed to fetch paragraph {paragraph_num}: {e}"
    
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
        """Answer a question about the book using hybrid retrieval and LLM generation."""
        if not query or not query.strip():
            return "Please provide a question about the book."
        
        query = query.strip()
        
        # Check cache first
        cached_result = self._get_cached_result(query)
        if cached_result:
            print("Retrieved answer from cache")
            return cached_result
        
        try:
            # Check for positional sentence queries using pre-loaded index (fastest)
            positional_response = self._detect_positional_sentence_query(query)
            if positional_response:
                self._cache_result(query, positional_response)
                return positional_response
            

            
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
            # TODO: Replace with dynamic token-aware truncation if needed
            formatted_prompt = self._format_retrieval_prompt(query, reranked_docs[:2])  # Use top 2
            
            # Debug output
            print(f"Top retrieved chunk: {reranked_docs[0].page_content[:100]}...")
            
            # Step 4: Generate answer
            result = self.llm.invoke(formatted_prompt).content
            
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

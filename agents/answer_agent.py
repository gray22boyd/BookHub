from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

class AnswerAgent:
    def __init__(self, book_title):
        self.book_title = book_title
        self.sanitized_title = book_title.lower().replace(" ", "_")
        
        # Load the FAISS index
        embeddings = OpenAIEmbeddings()
        index_path = f"faiss_indexes/{self.sanitized_title}"
        
        # Load the vector store
        self.vectorstore = FAISS.load_local(index_path, embeddings)
        
        # Create a retriever with k=5
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.2),
            chain_type="stuff",
            retriever=retriever
        )
    
    def answer(self, query: str) -> str:
        """
        Process a question and return an answer based on the book's content.
        
        Args:
            query (str): The question to answer
            
        Returns:
            str: The answer to the question
        """
        result = self.qa_chain.run(query)
        return result 
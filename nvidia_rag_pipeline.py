
"""
RAG Pipeline with NVIDIA NIM Models
Supports: PDF, DOCX, PPT, TXT files
Uses: NVIDIA Embeddings, Reranking, and LLM models via LangChain
"""

import os
from pathlib import Path
from typing import List

# Core LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# NVIDIA AI Endpoints
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)

# Text Splitter and Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class MultiFormatRAGPipeline:
    """RAG pipeline with NVIDIA models supporting multiple document formats"""

    def __init__(
        self,
        nvidia_api_key: str,
        embedding_model: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        llm_model: str = "meta/llama-3.3-70b-instruct",
        rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 8,
        top_n_rerank: int = 4
    ):
        """
        Initialize RAG pipeline with NVIDIA models

        Args:
            nvidia_api_key: NVIDIA API key from build.nvidia.com
            embedding_model: NVIDIA embedding model name
            llm_model: NVIDIA LLM model name
            rerank_model: NVIDIA reranking model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            top_n_rerank: Number of documents after reranking
        """
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key

        # Initialize NVIDIA models
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            truncate="END"
        )

        self.llm = ChatNVIDIA(
            model=llm_model,
            temperature=0.2,
            max_tokens=1024
        )

        self.reranker = NVIDIARerank(
            model=rerank_model,
            top_n=top_n_rerank,
            truncate="END"
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.top_k = top_k
        self.vectorstore = None
        self.rag_chain = None

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from multiple file formats

        Args:
            file_paths: List of file paths to load

        Returns:
            List of Document objects
        """
        documents = []

        for file_path in file_paths:
            file_ext = Path(file_path).suffix.lower()

            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_ext in ['.docx', '.doc']:
                    loader = Docx2txtLoader(file_path)
                elif file_ext in ['.pptx', '.ppt']:
                    loader = UnstructuredPowerPointLoader(file_path)
                elif file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    print(f"Unsupported file format: {file_ext} for {file_path}")
                    continue

                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents from {file_path}")

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        return documents

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
            documents: List of Document objects
        """
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")

        # Create FAISS vector store with NVIDIA embeddings
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        print("Vector store created successfully")

    def setup_rag_chain(self):
        """Setup the RAG chain with prompt template"""

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        # RAG prompt template
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the user's question accurately and concisely.

If the answer cannot be found in the context, clearly state that you don't have enough information to answer the question. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create retrieval chain with reranking
        def retrieve_and_rerank(query: str) -> str:
            """Retrieve documents and apply reranking"""
            # Initial retrieval
            docs = retriever.invoke(query)

            # Rerank documents
            reranked_docs = self.reranker.compress_documents(
                query=query,
                documents=docs
            )

            # Combine reranked document content
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            return context

        # Build RAG chain
        self.rag_chain = (
            {
                "context": lambda x: retrieve_and_rerank(x["question"]),
                "question": lambda x: x["question"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("RAG chain setup complete")

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline

        Args:
            question: User question

        Returns:
            Generated answer
        """
        if self.rag_chain is None:
            raise ValueError("RAG chain not initialized. Call setup_rag_chain() first.")

        response = self.rag_chain.invoke({"question": question})
        return response


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MultiFormatRAGPipeline(
        nvidia_api_key="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",  # Get from build.nvidia.com
        embedding_model="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        llm_model="meta/llama-3.3-70b-instruct",
        rerank_model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        top_n_rerank=3
    )

    # Load documents from multiple sources
    file_paths = [
        # r"D:\DATA\TalkwithDATA\Hymba_ICLR25.pdf",
        # r"D:\DATA\TalkwithDATA\AgmatelLeavePolicy.docx",
        r"D:\DATA\TalkwithDATA\Physical_AI.txt",
        # r"D:\DATA\TalkwithDATA\CES_Announcements.pptx"
    ]

    documents = pipeline.load_documents(file_paths)

    # Create vector store
    pipeline.create_vectorstore(documents)

    # Setup RAG chain
    pipeline.setup_rag_chain()

    # Query the pipeline
    # question = "What is major annocuments in Nvidas CES 2026 ?"
    # question2 = "What is told about leave policy ?"
    question = "what a"
    answer = pipeline.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

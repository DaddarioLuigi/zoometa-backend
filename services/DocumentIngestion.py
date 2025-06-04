from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import Document

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.storage import StorageContext
from llama_index.core.prompts import PromptTemplate

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

import pinecone
import os
from getpass import getpass
from transformers import GPT2TokenizerFast


class DocumentIngestion:
    def __init__(self, kb_dir, embed_model, vector_store):
        self.kb_dir = kb_dir
        self.embed_model = embed_model
        self.vector_store = vector_store

    def preprocess_documents(self, documents, chunk_size=4000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = []
        for doc in documents:
            split_docs.extend(text_splitter.split_text(doc.text))
        return [Document(text=t) for t in split_docs]

    def ingest_documents(self):
        # Load and preprocess documents
        documents = SimpleDirectoryReader(self.kb_dir).load_data()
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Ingest into vector store
        pipeline = IngestionPipeline(
            transformations=[self.embed_model],
            vector_store=self.vector_store
        )
        self.process_in_batches(pipeline, preprocessed_docs)

    def process_in_batches(self, pipeline, documents, batch_size=5):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            pipeline.run(documents=batch)

    def count_tokens(self, text):
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return len(tokenizer.encode(text))
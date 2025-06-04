from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline


from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

from llama_index.core import VectorStoreIndex,get_response_synthesizer,StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from langchain.text_splitter import RecursiveCharacterTextSplitter


from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

import pinecone

import os
from getpass import getpass
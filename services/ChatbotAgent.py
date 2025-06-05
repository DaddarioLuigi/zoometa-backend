from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.schema import Document

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
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

import os
import pinecone
import json
import logging
from getpass import getpass


class ChatbotAgent:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, informative_tool, recommendation_tool, llm, initial_context):
        self.agent = OpenAIAgent.from_tools(
            [informative_tool, recommendation_tool],
            llm=llm,
            verbose=True,
            system_prompt=initial_context,
            temperature=0.4,
        )
        self.chat_history = []

    def process_user_input(self, user_query):
        """Process a single query while keeping conversation context."""
        self.chat_history.append(f"User: {user_query}")

        # Let the agent manage the chat history internally
        response = self.agent.chat(user_query)

        logging.info("Finestra di contesto:\n%s", "\n".join(self.chat_history))

        text = getattr(response, "response", response)
        self.chat_history.append(f"Assistant: {text}")
        return response

from .PineconeManager import PineconeManager
from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .DocumentIngestion import DocumentIngestion
from .ChatbotAgent import ChatbotAgent

import re

class ChatService:
    def __init__(self, pinecone_api_key, openai_api_key, kb_dir, product_kb_dir):
        # Initialize LLM and Embedding settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)

        # Initialize Pinecone and OpenAI
        self.pinecone_manager = PineconeManager(pinecone_api_key)
        
        # Initialize vector stores
        self.vector_store = None
        self.vector_store_product = None
        self.init_pinecone("main-index", "product-index")

        # Initialize document ingestion for knowledge base and product KB
        self.document_ingestion = DocumentIngestion(kb_dir, embed_model=Settings.embed_model, vector_store=self.vector_store)
        self.product_ingestion = DocumentIngestion(product_kb_dir, embed_model=Settings.embed_model, vector_store=self.vector_store_product)
        
        # Initialize chatbot agent
        self.chatbot_agent = None
        self.init_agent()
    
    def reset_session(self):
        self.init_agent()

    def init_pinecone(self, main_index_name, product_index_name):
        # Check if the main index exists
        if not self.pinecone_manager.index_exists(main_index_name):
            self.pinecone_manager.create_index(main_index_name)
        
        if not self.pinecone_manager.index_exists(product_index_name):
            self.pinecone_manager.create_index(product_index_name)

        # Get the vector stores
        self.vector_store = self.pinecone_manager.get_vector_store(main_index_name)
        self.vector_store_product = self.pinecone_manager.get_vector_store(product_index_name)

        # Initialize vector indices
        self.vector_index_product = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store_product, 
            embed_model=Settings.embed_model
        )
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, 
            embed_model=Settings.embed_model
        )

    def ingest_knowledge_bases(self):
        self.document_ingestion.ingest_documents()
        self.product_ingestion.ingest_documents()

    def init_agent(self):
        informative_tool = QueryEngineTool(
            query_engine=self.vector_index.as_query_engine(),
            metadata=ToolMetadata(
                name="informative_tool",
                description=(
                    "Usa questo strumento per rispondere a domande di tipo informativo. "
                    "IMPORTANTE: Consiglia SOLO prodotti che sono specificatamente elencati nel database dei prodotti (vector_index_product). "
                    "NON inventare o suggerire prodotti che non sono presenti in vector_index_product."
                ),
            ),
        )

        recommendation_tool = QueryEngineTool(
            query_engine=self.vector_index_product.as_query_engine(),
            metadata=ToolMetadata(
                name="recommendation_tool",
                description=(
                    "Usa SEMPRE questo strumento per trovare e consigliare prodotti specifici in base alle esigenze dell'utente. "
                    "Fornisci dettagli sul prodotto come nome, caratteristiche principali e perché è adatto alle esigenze del cliente. "
                    "IMPORTANTE: Consiglia SOLO prodotti che sono specificatamente elencati nel database dei prodotti (vector_index_product). "
                    "NON inventare o suggerire prodotti che non sono presenti in questo database. "
                    "Se non trovi un prodotto adatto, comunica onestamente che non ci sono prodotti corrispondenti alle esigenze specificate. "
                    "Quando consigli un prodotto, fornisci il link del prodotto sul sito di ZooMeta SEMPRE in questo formato: "
                    "<a href='https://www.luxdada.it/zoometa/ricerca?controller=search&s=Nomeprodotto'>Nome prodotto</a>"
                ),
            ),
        )

        base_prompt = """[...il tuo prompt è perfetto e rimane invariato...]"""
        
        self.chatbot_agent = ChatbotAgent(
            informative_tool, 
            recommendation_tool, 
            llm=Settings.llm, 
            initial_context=base_prompt
        )

    def handle_user_query(self, user_query):
        response = self.chatbot_agent.process_user_input(user_query)
        return self.format_response_as_html(response.response)

    def format_response_as_html(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'###(.*?)\n', r'<strong>\1</strong><br>', text)
        text = text.replace('\n', '<br>')
        text = re.sub(r'^\s*-\s(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
        return text

from .PineconeManager import PineconeManager
from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from .CustomRecommendationTool import CustomRecommendationTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .DocumentIngestion import DocumentIngestion
from .ChatbotAgent import ChatbotAgent
import json

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
        """Ensure that Pinecone indices exist and are optimised."""
        dimension = 1536  # dimension for OpenAI embeddings

        if not self.pinecone_manager.index_exists(main_index_name):
            self.pinecone_manager.create_index(main_index_name, dimension=dimension)

        if not self.pinecone_manager.index_exists(product_index_name):
            self.pinecone_manager.create_index(product_index_name, dimension=dimension)

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

    def delete_indices(self, main_index_name, product_index_name):
        """Remove Pinecone indices if they exist."""
        self.pinecone_manager.delete_index(main_index_name)
        self.pinecone_manager.delete_index(product_index_name)

    def init_agent(self):
                # ──────────────────────────────────────────────────────────────────────────
        #  Informative tool  →  per risposte di contenuto (NO prodotti)
        # ──────────────────────────────────────────────────────────────────────────
        informative_tool = QueryEngineTool(
            query_engine=self.vector_index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="informative_tool",
                description=(
                    "Risponde a domande generali su missione ZooMeta, linee guida interne, "
                    "procedure di contatto o consigli veterinari generici. "
                    "Non menziona prodotti né link esterni."
                ),
            ),
        )

        # ──────────────────────────────────────────────────────────────────────────
        #  Recommendation tool  →  per cercare e consigliare prodotti
        # ──────────────────────────────────────────────────────────────────────────
        recommendation_tool = CustomRecommendationTool(
            query_engine=self.vector_index_product.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="recommendation_tool",
                description=(
                    "Restituisce sempre un JSON con fino a 3 prodotti pertinenti, "
                    "secondo lo schema:\n"
                    "{ 'products': [ { "
                    "'id_product', 'nome', 'descrizione_breve', 'descrizione', "
                    "'prezzo', 'produttore', 'giacenza', 'categorie', 'product_url' } ] }.\n"
                    "Se non ci sono prodotti idonei, restituisce 'products': [] nel JSON."
                    "Rispondi solo con il JSON puro, senza virgolette o testo aggiuntivo."
                ),
            ),
        )


        base_prompt = """
        Sei Arianna, l’assistente virtuale di ZooMeta specializzata in salute e nutrizione
        di cani, gatti e altri animali domestici.

        LINEE GUIDA GENERALI
        • Rispondi in modo empatico, professionale e rassicurante.
        • Se il messaggio dell’utente è offensivo (es. ‘troia’, ‘pompino’, ecc.):
        - Prima occorrenza -> “Per favore, utilizza un linguaggio rispettoso. Se hai bisogno di supporto reale per il tuo animale, sono qui per aiutarti.”
        - Recidiva -> “La conversazione è stata chiusa per linguaggio inappropriato. Per ulteriori necessità, contatta l’assistenza clienti.”
        - Non aggiungere altro.
        • Non dire mai barzellette.
        • Non menzionare negozi o veterinari esterni.

        WORKFLOW
        1. Saluto iniziale: «Ciao. Sono Arianna, assistente virtuale di ZooMeta. Come posso aiutarti oggi?»

        2. Raccolta dati (se necessari):
        - Nome animale, Specie/Razza, Età, Peso, Fase di vita (cucciolo, adulto, senior)
        - Taglia (toy, piccola, media, grande), Condizioni di salute rilevanti

        3. **Decisione formato:**
        - **Puremente informativa** → invoca **solo** `informative_tool` e restituisci **testo libero**.
        - **Altrimenti** → invoca **sempre** `recommendation_tool` e restituisci **un unico blocco JSON** con **fino a 3** prodotti, nel formato:
            ```json
            {
            "products": [
                {
                "id_product": "string",
                "nome": "string",
                "descrizione_breve": "string",
                "descrizione": "string",
                "prezzo": "string",
                "produttore": "string",
                "giacenza": integer,
                "categorie": ["string", "..."],
                "product_url": "string"
                }
                // max 3 elementi
            ]
            }
            ```
        - **Non** mescolare testi descrittivi e JSON: o l’uno o l’altro.

        4. Se la richiesta non riguarda prodotti ma solo informazioni generali o cliniche, usa `informative_tool`.  
        Se è una domanda clinica molto specifica, suggerisci di scrivere a info@zoometa.it.

        5. Conclusione standard (solo dopo `informative_tool`):
        «Spero di esserti stata utile. Per qualsiasi altra domanda sono qui. Se desideri un consulto più approfondito con i nostri specialisti ZooMeta, scrivici a info@zoometa.it. Grazie per aver scelto ZooMeta.»
        """
            
        self.chatbot_agent = ChatbotAgent(
            informative_tool, 
            recommendation_tool, 
            llm=Settings.llm, 
            initial_context=base_prompt
        )

    def handle_user_query(self, user_query, response_format="json"):
        """Handle a single user query and format the output."""
        response = self.chatbot_agent.process_user_input(user_query)
        text = getattr(response, "response", response)
        # LOG DI DEBUG: vedi in console cosa restituisce l'LLM
        print("DEBUG: raw LLM output →", text)

        if response_format == "json":
            return self.format_response_as_json(text)
        if response_format == "text":
            return text
        return self.format_response_as_html(text)

    def format_response_as_html(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'###(.*?)\n', r'<strong>\1</strong><br>', text)
        text = text.replace('\n', '<br>')
        text = re.sub(r'^\s*-\s(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
        return text

    def format_response_as_json(self, text):
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return text[start:end]
        except ValueError:
            return text
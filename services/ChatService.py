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
                    "Usa questo tool per rispondere a domande informative di carattere generale "
                    "sulla missione di ZooMeta, linee guida cliniche interne, procedure di contatto "
                    "o consigli veterinari generici **che NON richiedono la proposta di un prodotto**. "
                    "Non inventare né accennare a prodotti: per quelli esiste recommendation_tool."
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
                    "Invoca SEMPRE questo tool quando hai già raccolto tutte le informazioni chiave "
                    "e devi consigliare 1-3 articoli dal database prodotti (vector_index_product). "
                    "Restituisce per ciascun prodotto: id_product, nome, descrizione_breve, descrizione, "
                    "prezzo, produttore, giacenza, categorie, product_url. "
                    "► Consiglia SOLO articoli presenti nel database; NON inventare nulla. "
                    "► Se non esistono articoli idonei, comunica onestamente che non ci sono "
                    "prodotti compatibili e fornisci un link di ricerca generico, nel formato:\n"
                    "   <a href=\"https://www.zoometa.it/ricerca?controller=search&s=PAROLE+CHIAVE\">Cerca altri prodotti</a>\n"
                    "► Per ogni prodotto consigliato inserisci SEMPRE il link diretto:\n"
                    "   <a href=\"https://zoometa.it/index.php?controller=product&id_product=IDPRODOTTO\">Vedi prodotto</a>"
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
2. Raccolta dati se necessario (chiedili se mancano):
   - Nome animale - Specie/Razza - Età - Peso - Fase di vita (cucciolo, adulto, senior)
   - Taglia (toy, piccola, media, grande) - Condizioni di salute rilevanti
3. Se hai info sufficienti, puoi già proporre 1-3 prodotti:
   • Invoca `recommendation_tool` con query ben filtrata.
   • Verifica: specie corretta, fase di vita e taglia compatibili, giacenza > 0.
   • Dai priorità a prodotti con parole chiave: “bio”, “eco-friendly”, “riciclato”, ecc.
   • Formatta ciascun prodotto così:
      1. Nome prodotto
      2. Beneficio chiave (descrizione_breve o sintesi)
      3. Nota sostenibilità se presente
      4. Produttore
      5. Prezzo
      6.Link HTML (questo è un esempio): <a href="https://zoometa.it/index.php?controller=product&id_product=ID">Vedi prodotto</a>
   • Mai mescolare specie diverse. Mai consigliare prodotto non idoneo a taglia o fase vita.
   • Se non ci sono articoli idonei, fornisci link di ricerca generico:
     <a href="https://www.zoometa.it/ricerca?controller=search&s=PAROLE+CHIAVE">Cerca altri prodotti</a>
4. Per domande senza suggerimento di prodotti o di natura clinica generica:
   • Invoca `informative_tool`.
   • Se la richiesta è clinica molto specifica, invita a scrivere a info@zoometa.it (consulto interno).
5. Conclusione standard: «Spero di esserti stata utile. Per qualsiasi altra domanda sono qui. Se desideri un consulto più approfondito con i nostri specialisti ZooMeta, scrivici a info@zoometa.it. Grazie per aver scelto ZooMeta.»

ISTRUZIONI TECNICHE
• Se non sei certa della risposta, dichiara che la tua conoscenza è limitata alle informazioni di ZooMeta.
• Rispondi a eventuali domande multiple punto per punto.
• Non rispondere a domande non pertinenti agli animali domestici.
• Ogni volta che proponi, *devi* includere il link HTML appropriato.
• Non generare informazioni sui prodotti se il `recommendation_tool` non restituisce nulla.
"""
        
        self.chatbot_agent = ChatbotAgent(
            informative_tool, 
            recommendation_tool, 
            llm=Settings.llm, 
            initial_context=base_prompt
        )

    def handle_user_query(self, user_query, response_format="html"):
        """Handle a single user query and format the output."""
        response = self.chatbot_agent.process_user_input(user_query)
        text = getattr(response, "response", response)

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
        """Try to extract JSON from the LLM response."""
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return text[start:end]
        except ValueError:
            return text
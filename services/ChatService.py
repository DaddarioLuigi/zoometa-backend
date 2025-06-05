from .PineconeManager import PineconeManager
from llama_index.core.settings import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from .CustomRecommendationTool import CustomRecommendationTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .DocumentIngestion import DocumentIngestion
from .ChatbotAgent import ChatbotAgent
import openai
from gtts import gTTS
import base64
import uuid
import os
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
        informative_tool = QueryEngineTool(
            query_engine=self.vector_index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="informative_tool",
                description=(
                    "Usa questo strumento per rispondere a domande di tipo informativo. "
                    "IMPORTANTE: Consiglia SOLO prodotti che sono specificatamente elencati nel database dei prodotti (vector_index_product). "
                    "NON inventare o suggerire prodotti che non sono presenti in vector_index_product."
                ),
            ),
        )

        recommendation_tool = CustomRecommendationTool(
            query_engine=self.vector_index_product.as_query_engine(similarity_top_k=3),
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

        base_prompt = """
1. Chi è Arianna e qual è la sua missione
Arianna è l’assistente virtuale e veterinaria interna di ZooMeta.

Fornisce consigli completi su salute, nutrizione e benessere di cani, gatti e altri animali domestici e visualizza il file prodotti.csv (colonne: id_product, nome, descrizione, descrizione_breve, prezzo, product_url, produttore, giacenza, categorie) per proporre i prodotti più idonei alle esigenze dei clienti, dando priorità a quelli biologici, eco-friendly, riciclati o con packaging sostenibile.

Per richieste cliniche molto specifiche, invita l’utente a scrivere a info@zoometa.it per un consulto interno ZooMeta.
Non suggerisce mai di rivolgersi a un veterinario esterno.

2. Flusso di conversazione (Workflow)
Saluto iniziale
«Ciao. Sono Arianna,  assistente virtuale di ZooMeta. Come posso aiutarti oggi?»

Raccolta informazioni sull’animale
Chiedere sempre: Nome, Razza/specie, Età, Peso, Condizioni di salute rilevanti, Fase di vita (puppy, adulto, senior), Taglia (toy, piccola, media, grande)

Se la razza/specie è già chiara, Arianna può proporre subito almeno un prodotto pertinente, integrando nella stessa risposta eventuali domande mancanti.

Analisi dell’esigenza

-Per prodotti: chiedere (se necessario) preferenze su marca, gusto, formato.

-Per salute/comportamento: fornire il consiglio veterinario interno.

-Per approfondimenti: invitare a scrivere a info@zoometa.it.

3. Suggerimento di prodotti + link (regole imprescindibili)
Consultare prodotti.csv.

Filtrare:

-Specie corretta (mai confondere specie diverse; eccezione solo per multi-species)

-Fase di vita e taglia corretta

-Giacenza > 0 (non proporre prodotti esauriti)

Tra i risultati, dare priorità agli articoli con termini: “bio”, “biologico”, “organic”, “eco-friendly”, “riciclato”, “riciclabile”, “sostenibile”, “cruelty free”, “senza conservanti”, “senza additivi” presenti nel Nome Prodotto o nella Descrizione.

Per ciascun prodotto consigliato (massimo 3), includere:
Nome prodotto, Beneficio chiave (usare descrizione_breve, se disponibile, o sintesi di descrizione), Nota di sostenibilità (se applicabile), Produttore (es. «prodotto di [Marca]»), Prezzo (opzionale, formato “€ 12,99”)

Link HTML diretto (usare product_url oppure costruito con id_product)

Formato link obbligatorio:

<a href="https://zoometa.it/index.php?controller=product&id_product=IDPRODOTTO">Vedi prodotto</a>
Se non ci sono prodotti idonei, generare un link HTML di ricerca con parole chiave dell’utente:

<a href="https://www.zoometa.it/ricerca?controller=search&s=PAROLE+CHIAVE">Cerca altri prodotti</a>

Se l'utente richiede i consigli in formato JSON o la modalità JSON è attiva,
restituisci esclusivamente un oggetto nel formato:
{"products": [{"name": "Nome prodotto", "link": "URL", "brand": "Produttore", "price": "€ 0,00"}, ...]}
senza testo aggiuntivo.

4. Conclusione
«Spero di esserti stata utile. Per qualsiasi altra domanda sono qui. Se desideri un consulto più approfondito con i nostri specialisti ZooMeta, scrivici a info@zoometa.it. Grazie per aver scelto ZooMeta.»

5. Linee guida di stile e comportamento
Tono empatico, professionale, rassicurante.

- Non mischiare mai prodotti per specie diverse.
- Verificare sempre fase di vita e taglia.
- Evidenziare prodotti biologici, eco-friendly e sostenibili se disponibili.
- Mai menzionare concorrenti o veterinari esterni.
-Usare sempre i link in formato HTML.
-Integrare informazioni da descrizione_breve, produttore, product_url per arricchire i consigli.
-Non mischiare mai informazioni o consigli provenienti da prodotti differenti
-Se ricevi messaggi contenenti parole offensive, insulti, nonsense o provocazioni evidenti (es. “cazzo”, “banana” fuori contesto, “aiutami zio canr”,"troia, "pompino", "bocchini"), NON rispondere normalmente.
In questi casi limita la tua risposta a:
“Per favore, utilizza un linguaggio rispettoso. Se hai bisogno di supporto reale per il tuo animale, sono qui per aiutarti.”
Se il comportamento si ripete più volte nella stessa chat, chiudi educatamente la conversazione con:
“La conversazione è stata chiusa per linguaggio inappropriato. Per ulteriori necessità, contatta l’assistenza clienti
        """
        
        self.chatbot_agent = ChatbotAgent(
            informative_tool, 
            recommendation_tool, 
            llm=Settings.llm, 
            initial_context=base_prompt
        )

    def handle_user_query(self, user_query, response_format="html"):

    def transcribe_audio(self, audio_file):
        """Transcribe an uploaded audio file using OpenAI Whisper."""
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
        if isinstance(transcription, dict):
            return transcription.get("text", "")
        return getattr(transcription, "text", "")

    def text_to_speech(self, text):
        """Convert text to speech and return base64 encoded audio."""
        tts = gTTS(text=text, lang="it")
        tmp_path = f"/tmp/{uuid.uuid4().hex}.mp3"
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(tmp_path)
        return data

    def handle_audio_query(self, audio_file, response_format="html"):
        """Process an audio query and return text and audio."""
        user_text = self.transcribe_audio(audio_file)
        raw_response = self.handle_user_query(user_text, "text")

        if response_format == "json":
            formatted = self.format_response_as_json(raw_response)
        elif response_format == "text":
            formatted = raw_response
        else:
            formatted = self.format_response_as_html(raw_response)

        audio_base64 = self.text_to_speech(raw_response)
        return formatted, audio_base64
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
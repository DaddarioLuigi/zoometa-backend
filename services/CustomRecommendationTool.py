from llama_index.core.tools import QueryEngineTool, ToolMetadata
import json

class CustomRecommendationTool(QueryEngineTool):
    def __init__(self, query_engine, metadata):
        super().__init__(query_engine, metadata)

    def run(self, *args, **kwargs):
        # Esegui il tool e ottieni la risposta
        response = super().run(*args, **kwargs)
        
        # Gestisci la risposta per restituirla in un formato JSON
        return self.format_response(response)

    def format_response(self, response):
        # Assicurati che la risposta sia in formato JSON
        return json.dumps({"products": response})  # Modifica qui per includere i prodotti
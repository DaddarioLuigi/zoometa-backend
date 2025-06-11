from llama_index.core.tools import QueryEngineTool, ToolMetadata
import json

class CustomRecommendationTool(QueryEngineTool):
    def __init__(self, query_engine, metadata):
        super().__init__(query_engine, metadata)

    def run(self, *args, **kwargs):
        response = super().run(*args, **kwargs)
        return self._extract_json(response.response if hasattr(response, "response") else response)

    def _extract_json(self, text: str) -> str:
        # regex per oggetto JSON più esterno (richiede Python 3.11+)
        import re, json
        m = re.search(r'(\{(?:[^{}]|(?1))*\})', text, flags=re.DOTALL)
        if m:
            try:
                # riconosciuta struttura valida
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict) and "products" in parsed:
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        # fallback: nessun JSON valido, rispondo con lista vuota
        return json.dumps({"products": []})

from llama_index.core.tools import QueryEngineTool, ToolMetadata
import json

class CustomRecommendationTool(QueryEngineTool):
    def __init__(self, query_engine, metadata):
        super().__init__(query_engine, metadata)

    def run(self, *args, **kwargs):
        response = super().run(*args, **kwargs)
        return self.format_response(response)

    def format_response(self, response):
        text = getattr(response, "response", response)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "products" in parsed:
                return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
        return json.dumps({"products": text})

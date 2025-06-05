from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

class PineconeManager:
    def __init__(self, pinecone_api_key):
        # Create an instance of Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)

    def create_index(self, index_name, dimension=1536, cloud="aws", region="us-east-1", index_type='serverless'):
        # Create a Pinecone index using the instance
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec={
                    index_type: {
                        'cloud': cloud,
                        'region': region
                    }
                }
            )

    def get_vector_store(self, index_name):
        # Connect to an existing Pinecone index and return the vector store
        index = self.pc.Index(index_name)
        return PineconeVectorStore(pinecone_index=index)

    def index_exists(self, index_name):
        return index_name in self.pc.list_indexes().names()

    def delete_index(self, index_name):
        """Delete an index if it exists."""
        if self.index_exists(index_name):
            self.pc.delete_index(index_name)

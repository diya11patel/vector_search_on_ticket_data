from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to local instance
class QdrantConn:
    client: QdrantClient

    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333, timeout=10000)
        
    
    def create_db(self):
        print("Creating collection")
        self.client.recreate_collection(
        collection_name="support_tickets",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE)  # use actual size of your embedding
    )
        self.client.upsert
        return self.client
    
    
qclient = QdrantConn()

    




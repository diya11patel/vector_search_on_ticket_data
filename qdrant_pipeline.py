from typing import Any, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import uuid # For unique IDs if not provided

# Assuming db_connection.py is in the same directory and qclient is initialized there
from db_connections.qdrant import qclient

# --- Configuration ---
COLLECTION_NAME = "support_tickets_pipeline"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Common model, 384 dimensions
EMBEDDING_DIM = 384  # Must match the chosen SentenceTransformer model
N_CLUSTERS = 2  # Example: 2 clusters, as per "pre  defined k means cluster" (number of clusters)

# --- Sample Data ---
sample_tickets = [
    {"id": 1, "text": "Cannot connect to the company VPN, it keeps saying authentication failed.", "category": "Network"},
    {"id": 2, "text": "My Microsoft Outlook is not opening, it crashes on startup.", "category": "Software"},
    {"id": 3, "text": "The new laptop I received won't boot up, it shows a black screen.", "category": "Hardware"},
    {"id": 4, "text": "I'm unable to access the shared drive, it says permission denied.", "category": "Network"},
    {"id": 5, "text": "Excel is freezing frequently when I work with large spreadsheets.", "category": "Software"},
    {"id": 6, "text": "The printer on the 3rd floor is not working, documents are stuck in queue.", "category": "Hardware"},
    {"id": 7, "text": "How do I reset my domain password?", "category": "Account"},
]

# --- Qdrant Setup ---
def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Creates a Qdrant collection if it doesn't already exist.
    """
    import pdb; pdb.set_trace()
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # A more specific exception would be better, e.g., from qdrant_client.http.exceptions
        print(f"Collection '{collection_name}' not found (Reason: {e}). Creating now.")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created with vector size {vector_size}.")

# --- Processing and Upserting ---
def process_and_upsert_tickets(
    tickets_data: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    kmeans_model: KMeans
):
    """
    Embeds ticket texts, clusters them, and upserts them into Qdrant.
    """
    texts = [ticket["text"] for ticket in tickets_data]
    import pdb; pdb.set_trace()
    print("Embedding texts...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    print("Clustering embeddings...")
    # For a truly "pre-defined" (i.e., already trained) KMeans, you would load it
    # and use kmeans_model.predict(embeddings).
    # Here, we fit it to the current batch of tickets for demonstration.
    cluster_ids = kmeans_model.fit_predict(embeddings)
    
    points_to_upsert = []
    for i, ticket in enumerate(tickets_data):
        # Use provided ID or generate a UUID if not present
        point_id = ticket.get("id", str(uuid.uuid4())) 
        
        payload = {
            "text": ticket["text"],
            "category": ticket.get("category", "Unknown"), # Handle missing category
            "cluster_id": int(cluster_ids[i]) # Ensure cluster_id is JSON serializable (int)
        }
        points_to_upsert.append(
            PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(), # Convert numpy array to list
                payload=payload
            )
        )
            
    print(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{COLLECTION_NAME}'...")
    qclient.client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_to_upsert,
        wait=True  # Wait for operation to complete
    )
    print("Upsert complete.")

    # Verify by scrolling a few items
    print("\nVerifying a few data points in Qdrant:")
    try:
        data_verification = qclient.client.scroll(collection_name=COLLECTION_NAME, limit=5, with_vectors=False)[0]
        if data_verification:
            for record in data_verification:
                print(f"  ID: {record.id}, Payload: {record.payload}")
        else:
            print("  No data found in collection for verification scroll.")
    except Exception as e:
        print(f"  Error during verification scroll: {e}")

# --- Querying ---
def query_similar_tickets(
    query_text: str,
    embedding_model: SentenceTransformer,
    top_k: int = 3
):
    """
    Embeds a query text and searches for similar tickets in Qdrant.
    """
    print(f"\nQuerying for: '{query_text}'")
    query_vector = embedding_model.encode([query_text])[0].tolist()
    
    try:
        import pdb; pdb.set_trace()
        search_results = qclient.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True # Ensure payload is returned for context
        )
    except Exception as e:
        print(f"Error during search: {e}")
        return

    print("Search Results:")
    if not search_results:
        print("No similar tickets found.")
        return

    for i, hit in enumerate(search_results):
        print(f"  Rank {i+1}:")
        print(f"    ID: {hit.id}")
        print(f"    Score: {hit.score:.4f}")
        print(f"    Text: {hit.payload.get('text', 'N/A')}")
        print(f"    Category: {hit.payload.get('category', 'N/A')}")
        print(f"    Cluster ID: {hit.payload.get('cluster_id', 'N/A')}")
        
        # Placeholder for passing to Gemini
        print(f"    Full Payload (for potential Gemini input): {hit.payload}")
        # Example: sop = generate_sop_with_gemini(hit.payload['text'], hit.payload)
        print("-" * 20)

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize models
    print("Loading sentence embedding model...")
    sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    import pdb; pdb.set_trace()
    print(f"Initializing KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')

    # Setup Qdrant collection
    # This uses the qclient from db_connection.py
    setup_qdrant_collection(qclient.client, COLLECTION_NAME, EMBEDDING_DIM)
    
    # Process and store tickets
    process_and_upsert_tickets(sample_tickets, sbert_model, kmeans)
    
    # Example Queries
    print("\n" + "="*30 + " RUNNING QUERIES " + "="*30)
    query_similar_tickets(
        "My computer is very slow and applications are crashing often.", 
        sbert_model, 
        top_k=2
    )
    
    query_similar_tickets(
        "vpn connection issue, cannot authenticate", 
        sbert_model, 
        top_k=2
    )

    query_similar_tickets(
        "password reset request",
        sbert_model,
        top_k=2
    )

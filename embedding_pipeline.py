from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from db_connections.qdrant import session, Ticket
import numpy as np

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
N_CLUSTERS = 2

sample_tickets = [
    {"ticket_id": "1", "text": "Cannot connect to the company VPN, it keeps saying authentication failed.", "category": "Network"},
    {"ticket_id": "2", "text": "My Microsoft Outlook is not opening, it crashes on startup.", "category": "Software"},
    {"ticket_id": "3", "text": "The new laptop I received won't boot up, it shows a black screen.", "category": "Hardware"},
    {"ticket_id": "4", "text": "I'm unable to access the shared drive, it says permission denied.", "category": "Network"},
    {"ticket_id": "5", "text": "Excel is freezing frequently when I work with large spreadsheets.", "category": "Software"},
    {"ticket_id": "6", "text": "The printer on the 3rd floor is not working, documents are stuck in queue.", "category": "Hardware"},
    {"ticket_id": "7", "text": "How do I reset my domain password?", "category": "Account"},
]

def process_and_store_to_pgvector(tickets: List[Dict[str, Any]], model: SentenceTransformer, kmeans: KMeans):
    texts = [ticket['text'] for ticket in tickets]
    embeddings = model.encode(texts, show_progress_bar=True)
    cluster_ids = kmeans.fit_predict(embeddings)

    for i, ticket in enumerate(tickets):
        embedding = np.array(embeddings[i], dtype=np.float32)
        new_ticket = Ticket(
            ticket_id=ticket['ticket_id'],
            text=ticket['text'],
            embedding=embedding
        )
        session.add(new_ticket)

    session.commit()
    print(f"Inserted {len(tickets)} tickets into PostgreSQL with pgvector.")

def query_similar_from_pgvector(query: str, model: SentenceTransformer, top_k: int = 3):
    query_vector = np.array(model.encode([query])[0], dtype=np.float32)

    sql = f"""
    SELECT id, ticket_id, text,
           embedding <#> cube(:query_embedding) as distance
    FROM tickets
    ORDER BY embedding <#> cube(:query_embedding)
    LIMIT :top_k;
    """

    result = session.execute(
        sql,
        {"query_embedding": query_vector.tolist(), "top_k": top_k}
    )

    print("Search Results:")
    for row in result:
        print(f"ID: {row.id}, Ticket ID: {row.ticket_id}, Text: {row.text}, Distance: {row.distance:.4f}")

if __name__ == "__main__":
    sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')

    process_and_store_to_pgvector(sample_tickets, sbert_model, kmeans)

    print("\n--- Query Examples ---")
    query_similar_from_pgvector("vpn authentication failed", sbert_model)
    query_similar_from_pgvector("Excel crashing", sbert_model)

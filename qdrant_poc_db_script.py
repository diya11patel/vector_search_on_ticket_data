from typing import Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from db_connections.qdrant import qclient
from qdrant_client.models import PointStruct
import numpy as np


# Sample tickets
tickets = [
    {"id": 1, "text": "Cannot connect to VPN", "category": "Network"},
    {"id": 2, "text": "Outlook not opening", "category": "Software"},
    {"id": 3, "text": "Laptop won't boot", "category": "Hardware"},
]

vectors = np.array([
    [0.1, 0.3, 0.7],
    [0.2, 0.4, 0.6],
    [0.9, 0.1, 0.1]
])

def upsert_data(tickets: list[Any]):
    points = [
        PointStruct(id=ticket["id"], vector=vec.tolist(), payload=ticket)
        for ticket, vec in zip(tickets, vectors)
    ]

    print(vectors)
    print(points)
    # qclient.client.upsert(collection_name="support_tickets", points=points)
    data = qclient.client.scroll(collection_name="support_tickets", limit=100, with_vectors=True)[0]

    print(data)


def QueryTikcets(query_vector: List[float]):
    results = qclient.client.search(
    collection_name="support_tickets",
    query_vector=query_vector,
    limit=3  # Top 3 most similar tickets
    )
    for res in results:
        print(f"Match ID: {res.id}, Score: {res.score}, Text: {res.payload['text']}")


QueryTikcets([0.1, 0.2, 0.6])

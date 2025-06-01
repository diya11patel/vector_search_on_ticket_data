from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Setup DB
engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")
Base = declarative_base()

# 2. Define Table
class Ticket(Base):
    __tablename__ = 'tickets'

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String)
    text = Column(String)
    embedding = Column(Vector(768))  # or 768 based on your model
    cluster_id = Column(Integer) 

Base.metadata.reflect(bind=engine)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()



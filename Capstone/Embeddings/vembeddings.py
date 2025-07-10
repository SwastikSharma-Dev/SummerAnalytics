import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Load CSV
df = pd.read_csv("Embeddings\questions_tagged_together.csv")

# Combine columns into meaningful context
def row_to_doc(row):
    metadata = {
        "company": row["company"],
        "role": row["role"],
        "experience_level": row["experience_level"],
        "round": row["round"],
        "topic": row["topic"],
        "difficulty": row["difficulty"],
        "source_link": row["source_link"]
    }
    return Document(page_content=row["question"], metadata=metadata)

docs = [row_to_doc(row) for _, row in df.iterrows()]

# Embed using sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS DB
vector_db = FAISS.from_documents(docs, embeddings)
vector_db.save_local("company_qa_vector_db")

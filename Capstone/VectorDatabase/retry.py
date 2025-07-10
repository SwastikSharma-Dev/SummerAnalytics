from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

# # Load CSV
# df = pd.read_csv("Embeddings\questions_tagged_together.csv")

# # Create documents with embedded metadata
# documents = [
#     Document(
#         page_content=f"Company: {row['company']}\nRole: {row['role']}\nExperience: {row['experience_level']}\nRound: {row['round']}\nTopic: {row['topic']}\nDifficulty: {row['difficulty']}\nQuestion: {row['question']}",
#         metadata=row.to_dict()
#     )
#     for _, row in df.iterrows()
# ]

# # Initialize embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create FAISS vector store
# db = FAISS.from_documents(documents, embedding_model)

# # Save the DB
# db.save_local("company_vector_db")
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# 1. Load and prepare CSV
csv_path = "Embeddings\questions_tagged_together.csv"  # Update path
df = pd.read_csv(csv_path)

# 2. Combine relevant fields
df["content"] = df.apply(lambda row: f"{row['company']} | {row['role']} | {row['experience_level']} | {row['question']}", axis=1)

# 3. Create document objects
docs = [Document(page_content=row["content"], metadata=row.to_dict()) for _, row in df.iterrows()]

# 4. Create or load vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("company_qa_vector_db")

# 5. Load vectorstore for retrieval
retriever = FAISS.load_local("company_qa_vector_db", embedding, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 5})

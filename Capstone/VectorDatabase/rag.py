from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# === CONFIG ===
VECTOR_DB_DIR = "db_question_vectors"
COLLECTION_NAME = "interview_questions"

# === Init DB and Model ===
client = PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Search Function ===
def query_vector_db(user_query, k=5):
    query_embedding = model.encode([user_query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    for i in range(len(results['documents'][0])):
        print(f"\n🔹 Result #{i+1}")
        print(f"Question: {results['documents'][0][i]}")
        print(f"Metadata: {results['metadatas'][0][i]}")

    return results['documents'][0], results['metadatas'][0]

from together import Together

client = Together(api_key="CHANGED HAHAHAHA WILL USE ENV LATER")  

def ask_rag_bot(user_query):
    docs, _ = query_vector_db(user_query)

    context = "\n".join(docs)

    prompt = f"""Use the following past questions to generate a preparation guide for the query:
---
{context}
---
Query: {user_query}
Answer:"""

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\n🧠 Generated Response:\n")
    print(response.choices[0].message.content)
    return response.choices[0].message.content


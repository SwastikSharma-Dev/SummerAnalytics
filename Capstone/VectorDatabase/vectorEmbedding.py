import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "questions_tagged_together.csv"
VECTOR_DB_DIR = "db_question_vectors"
COLLECTION_NAME = "interview_questions"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["question"], inplace=True)

# === Generate Embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Generating embeddings...")

questions = df["question"].tolist()
embeddings = model.encode(questions, show_progress_bar=True)

# === NEW WAY TO INIT CHROMA ===
client = PersistentClient(path=VECTOR_DB_DIR)

# Check if collection already exists
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(name=COLLECTION_NAME)

# === Add documents with metadata ===
print("Inserting into ChromaDB...")
collection.add(
    documents=questions,
    embeddings=embeddings,
    metadatas=df.to_dict("records"),
    ids=[f"id_{i}" for i in range(len(questions))]
)

print("✅ Vector DB created at:", VECTOR_DB_DIR)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser

# 1. Load embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load vector database
vector_db = FAISS.load_local("company_qa_vector_db", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 3. Load LLM
llm = ChatOllama(model="mistral:latest")

# 4. Prompt template
prompt_template = PromptTemplate.from_template("""
You are an interview assistant. Answer the user's question based on the following context. 
If no relevant information is found, reply with: "I do not have knowledge on this. "

Company: {company}
Role: {role}
Experience: {experience}
Context: {context}
Question: {question}

Answer:
""")

def fallback_if_empty(docs):
    if not docs:
        return [{"page_content": "I do not have knowledge on this. "}]
    return docs

rag_chain = RunnableMap({
    "context": RunnableLambda(lambda x: f"Company: {x['company']}\nRole: {x['role']}\nExperience: {x['experience']}\nQuestion: {x['question']}") | retriever | RunnableLambda(fallback_if_empty),
    "company": lambda x: x["company"],
    "role": lambda x: x["role"],
    "experience": lambda x: x["experience"],
    "question": lambda x: x["question"],
}) | prompt_template | llm | StrOutputParser()



# 6. Sample Input
inputs = {
    "company": "Google",
    "role": "Software Engineer",
    "experience": "New Grad",
    "question": "What kind of questions are asked in coding rounds? Give me the exact questions and topic. I want crisp and exact topics and questions. And also leetcode questions to practise on same topic."
}


# 7. Get Response
response = rag_chain.invoke(inputs)
print("Answer:", response)


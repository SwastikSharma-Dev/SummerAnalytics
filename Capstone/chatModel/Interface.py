# interface.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda

# 🔧 Must be the first Streamlit command
st.set_page_config(page_title="Interview Assistant | JobOverflow", layout="centered")

# 🚀 Load retriever once and cache it
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("../company_qa_vector_db", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_retriever()
llm = ChatOllama(model="mistral:latest")

# 🧠 Prompt Template
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

# 🔍 Handle empty docs
def fallback_if_empty(docs):
    if not docs:
        return [{"page_content": "I do not have knowledge on this. "}]
    return docs

# 🔗 RAG Chain
rag_chain = (
    RunnableMap({
        "context": RunnableLambda(lambda x: f"Company: {x['company']}\nRole: {x['role']}\nExperience: {x['experience']}\nQuestion: {x['question']}") 
                  | retriever 
                  | RunnableLambda(fallback_if_empty),
        "company": lambda x: x["company"],
        "role": lambda x: x["role"],
        "experience": lambda x: x["experience"],
        "question": lambda x: x["question"],
    }) 
    | prompt_template 
    | llm 
    | StrOutputParser()
)

# 🎨 UI
st.title("💼 Interview Assistant | by: Swastik | SA Capstone ")
st.markdown("Get interview questions and topics tailored to company, role, and experience.")

with st.form("query_form"):
    company = st.text_input("Company", placeholder="e.g., Google")
    role = st.text_input("Role", placeholder="e.g., Software Engineer")
    experience = st.selectbox("Experience Level", ["New Grad", "Intern", "0-1 years", "1-2 years", "2-5 years", "5+ years"])
    question = st.text_area("Your Question", placeholder="e.g., What kind of questions are asked in coding rounds?")
    submitted = st.form_submit_button("Get Answer")

if submitted:
    if not all([company, role, experience, question]):
        st.warning("Please fill out all fields.")
    else:
        with st.spinner("Searching database and generating response..."):
            query = {
                "company": company,
                "role": role,
                "experience": experience,
                "question": question
            }
            response = rag_chain.invoke(query)
            st.markdown("### 📋 Answer")
            st.success(response)

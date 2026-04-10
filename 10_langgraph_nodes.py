import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 0. Load the secret keys from the .env file!
load_dotenv()

# 1. Bring in our State (The Conveyor Belt)
class GraphState(TypedDict):
    question: str
    role: str
    documents: List[Document]
    answer: str

# 2. Re-connect to your secured database
DB_DIR = "vector_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 3. Initialize Gemini (It will now automatically find the key)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- DEFINE THE NODES (THE WORKERS) ---

def retrieve(state: GraphState):
    print("--- ACTIVATING NODE: Retriever ---")
    question = state["question"]
    role = state["role"]
    
    docs = vector_db.similarity_search(question, k=2, filter={"role": role})
    return {"documents": docs}

def fallback(state: GraphState):
    print("--- ACTIVATING NODE: Fallback Bouncer ---")
    safe_message = "I'm sorry, I either do not have information on this topic, or you do not have the required security clearance to view it."
    return {"answer": safe_message}

def generate(state: GraphState):
    print("--- ACTIVATING NODE: Generator ---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(doc.page_content for doc in documents)
    
    template = """You are a helpful enterprise AI assistant. 
    Use the following context to answer the user's question. 
    If the answer is not in the context, politely say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    return {"answer": response.content}

print("Success! The Retrieval, Fallback, and Generation nodes are locked and loaded.")
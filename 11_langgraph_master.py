from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# --- SETUP ---
# Load the re-ranking model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
load_dotenv()
DB_DIR = "vector_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- 1. THE STATE ---
class GraphState(TypedDict):
    question: str
    role: str
    documents: List[Document]
    answer: str

# --- 2. THE NODES (Workers) ---
def retrieve(state: GraphState):
    print("\n[Node] Retriever: Searching database...")
    
    # 1. Cast a wider net (k=5) using the fast Bi-Encoder
    initial_results = vector_db.similarity_search_with_score(
        state["question"], 
        k=5, 
        filter={"role": state["role"]}
    )
    
    # If the RBAC firewall blocks everything, return empty
    if not initial_results:
        return {"documents": []}
        
    # Extract just the documents from the initial search
    docs = [doc for doc, _ in initial_results]
    
    # 2. Format the data for the Cross-Encoder (Fixing the "Formatting Error")
    # It requires a list of pairs: [[Question, Document Text], [Question, Document Text]]
    pairs = [[state["question"], doc.page_content] for doc in docs]
    
    # 3. Score the pairs
    scores = cross_encoder.predict(pairs)
    
    # 4. Pair the new scores with the documents and sort them (highest score first)
    scored_docs = list(zip(scores, docs))
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    valid_docs = []
    
    # 5. Evaluate the absolute best document
    best_score, best_doc = sorted_docs[0]
    print(f"   -> Top Cross-Encoder Score: {best_score:.4f}")
    
    # Cross-Encoder scores are logits. Anything > 0 is generally a good match.
    if best_score > 0: 
        valid_docs.append(best_doc)
        print("   -> PASSED: Document verified by Cross-Encoder.")
    else:
        print("   -> REJECTED: Cross-Encoder deemed top document irrelevant.")
            
    return {"documents": valid_docs}

def fallback(state: GraphState):
    print("[Node] Fallback: Engaging safe response...")
    return {"answer": "I am sorry, but I either do not have information on this topic, or your current security clearance does not permit me to share it."}

def generate(state: GraphState):
    print("[Node] Generator: Drafting response with Gemini...")
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    template = """You are a helpful enterprise AI assistant. 
    Use the following context to answer the user's question. 
    If the answer is not in the context, politely say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}

# --- 3. THE EDGES (Traffic Cop) ---
def route_question(state: GraphState):
    print("[Routing] Traffic Cop: Checking for documents...")
    if len(state["documents"]) == 0:
        print(" -> Action: No documents found. Routing to Fallback.")
        return "fallback"
    else:
        print(" -> Action: Documents found. Routing to Generator.")
        return "generate"

# --- 4. BUILD THE ASSEMBLY LINE ---
print("Assembling the LangGraph State Machine...")
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("fallback", fallback)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", route_question)
workflow.add_edge("fallback", END)
workflow.add_edge("generate", END)

app = workflow.compile()
print("LangGraph compilation successful!\n")

# --- 5. THE ULTIMATE TEST ---
test_question = "What is the mandatory rotation frequency for Admin passwords?"

print("==================================================")
print("TEST 1: STANDARD EMPLOYEE (Should be blocked gracefully)")
employee_input = {"question": test_question, "role": "employee"}
employee_result = app.invoke(employee_input)
print(f"\nFINAL AI ANSWER: {employee_result['answer']}")
print("==================================================\n")

print("==================================================")
print("TEST 2: IT ADMIN (Should get the exact answer)")
admin_input = {"question": test_question, "role": "admin"}
admin_result = app.invoke(admin_input)
print(f"\nFINAL AI ANSWER: {admin_result['answer']}")
print("==================================================")
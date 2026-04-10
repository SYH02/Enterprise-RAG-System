# Enterprise RAG System with Role-Based Access Control (RBAC)

## Overview
This repository contains a secure, enterprise-grade Retrieval-Augmented Generation (RAG) pipeline. The system is designed to intelligently retrieve internal company documentation and answer user queries while strictly enforcing Role-Based Access Control (RBAC) to prevent unauthorized data exposure.

## System Architecture
The application utilizes a sophisticated state machine to orchestrate the retrieval and generation process, ensuring high accuracy and minimal hallucination.

* **Frontend:** Streamlit Community Cloud
* **Orchestration:** LangGraph (State Machine)
* **Vector Database:** ChromaDB 
* **Large Language Model:** Google Gemini 1.5/2.5 Flash
* **Embeddings:** HuggingFace Sentence Transformers
* **Re-ranking & Verification:** MS-MARCO Cross-Encoder

## Key Features
* **Strict RBAC Enforcement:** Metadata filtering at the database level ensures users can only query documents cleared for their specific security tier (Standard Employee, Finance, IT Admin).
* **Cross-Encoder Guardrails:** A semantic verification layer actively blocks the LLM from generating answers if the retrieved context does not directly match the user's query, effectively eliminating hallucinations.
* **Fallback Logic:** Queries outside the scope of the database or beyond the user's clearance are safely routed to a standardized fallback response.
* **Evaluated Baseline:** The retrieval accuracy (Context Precision) and hallucination resistance (Faithfulness) were mathematically evaluated using the Ragas grading framework.

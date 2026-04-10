import streamlit as st
import importlib

# 1. We must use importlib because Python doesn't like importing files that start with numbers
backend = importlib.import_module("11_langgraph_master")
rag_app = backend.app

# 2. Build the visual shell
st.title("Secure Enterprise RAG Assistant")

# 3. Build the RBAC Dropdown in a sidebar
st.sidebar.header("Security Settings")
user_role = st.sidebar.selectbox(
    "Select your active security clearance:",
    ["employee", "finance", "admin"]
)

# 4. Initialize the Chat Memory (st.session_state)
if "messages" not in st.session_state:
    # If no memory exists yet, create an empty list to hold the chat history
    st.session_state.messages = []

# 5. Render the historical messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Create the user text input box
if prompt := st.chat_input("Ask a question about the company policies..."):
    
    # Instantly draw the user's message on screen and save it to memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 7. Bridge to LangGraph & fix the "Serialization Error"
    with st.chat_message("assistant"):
        with st.spinner("Searching secure database..."):
            
            # We package the exact dictionary format LangGraph expects
            inputs = {"question": prompt, "role": user_role}
            
            # We invoke the LangGraph brain
            result = rag_app.invoke(inputs)
            
            # THE FIX: We ONLY extract the string text from the complex LangGraph state object.
            # Storing the raw object causes the serialization crash you reported to John.
            final_answer = result["answer"]
            
            # Draw the answer and save it to memory
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
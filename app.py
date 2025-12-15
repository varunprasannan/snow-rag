import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

# --- Imports (Strictly following your working setup) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader

# !!! CORRECTED IMPORTS FOR DEC 2025 STANDARD !!!
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = "support-agent-v1"
CSV_FILE_PATH = "1.csv"

st.set_page_config(page_title="Azure Support Agent", layout="wide")

# --- INITIALIZATION (Cached) ---
@st.cache_resource
def get_rag_chain():
    """
    Initializes the RAG chain using langchain_classic components.
    """
    # 1. Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Pinecone Vector Store
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # 4. Prompt
    system_prompt = (
        "You are an expert L3 Support Engineer for Azure Synapse and Data Factory."
        "\n\n"
        "CONTEXT: You are currently troubleshooting the following active ticket:\n"
        "{ticket_context}"
        "\n\n"
        "REFERENCE: Use the following retrieved resolved tickets to help solve the active ticket."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build Chain (Using langchain_classic)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def get_ticket_details(ticket_id):
    """Fetches ticket details from the local CSV."""
    if not os.path.exists(CSV_FILE_PATH):
        return None, "CSV file not found."
    
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        # Ensure string comparison
        ticket = df[df['ticket_id'].astype(str) == str(ticket_id)]
        
        if ticket.empty:
            return None, "Ticket ID not found."
        
        return ticket.iloc[0].to_dict(), None
    except Exception as e:
        return None, str(e)

# --- SESSION STATE SETUP ---
if "ticket_locked" not in st.session_state:
    st.session_state.ticket_locked = False
if "current_ticket" not in st.session_state:
    st.session_state.current_ticket = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI LOGIC ---

# SCREEN 1: TICKET ENTRY (Forced First Step)
if not st.session_state.ticket_locked:
    st.title("🔐 Agent Workspace")
    st.markdown("### Enter Ticket ID to Begin")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        ticket_input = st.text_input("Ticket Number", placeholder="e.g. INC001")
        
        if st.button("Load Ticket"):
            if ticket_input:
                details, error = get_ticket_details(ticket_input)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.ticket_locked = True
                    st.session_state.current_ticket = details
                    # Add initial bot greeting
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": f"I've loaded **{ticket_input}**. \n\n**Issue:** {details.get('description')}\n\nI'm ready to help. What should we check first?"
                    }]
                    st.rerun()
            else:
                st.warning("Please enter a ticket ID.")

# SCREEN 2: CHAT INTERFACE
else:
    # Sidebar with Ticket Info
    with st.sidebar:
        st.header("📋 Active Ticket")
        ticket = st.session_state.current_ticket
        st.info(f"**{ticket.get('ticket_id')}**")
        st.write(f"**Status:** {ticket.get('status', 'Unknown')}")
        st.write(f"**Priority:** {ticket.get('priority', 'Unknown')}")
        st.markdown("---")
        st.caption("Description")
        st.write(ticket.get('description', 'No description'))
        
        if st.button("🔄 Change Ticket"):
            st.session_state.ticket_locked = False
            st.session_state.current_ticket = {}
            st.session_state.messages = []
            st.rerun()

    st.title("💬 Support Copilot")
    
    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Area
    if user_input := st.chat_input("Ask about this ticket..."):
        # Show User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    chain = get_rag_chain()
                    
                    # Pass context dynamically
                    ticket_context = f"ID: {ticket.get('ticket_id')}\nDesc: {ticket.get('description')}"
                    
                    response = chain.invoke({
                        "input": user_input,
                        "ticket_context": ticket_context
                    })
                    
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
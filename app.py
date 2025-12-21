# This app will let you directly ask a question regarding your Knowledge Base

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# ENV & CONFIG
# --------------------------------------------------
load_dotenv()

INDEX_NAME = "support-agent-v1"
TOP_K = 3

# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="L3 Support AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 L3 Support AI Agent")
st.caption("Azure Synapse • Azure Data Factory • SeeQ")

# --------------------------------------------------
# CACHE MODELS (IMPORTANT FOR PERFORMANCE)
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

# --------------------------------------------------
# LOAD COMPONENTS
# --------------------------------------------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = load_llm()

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
system_prompt = (
    "You are an expert L3 Support Engineer for Azure Synapse, "
    "Azure Data Factory, and SeeQ.\n\n"
    "Use the following resolved support tickets to answer the question.\n"
    "Always cite the ticket IDs you used.\n"
    "If the answer is not present in the tickets, say that you don't know.\n\n"
    "Resolved Tickets:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# --------------------------------------------------
# CHAINS
# --------------------------------------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --------------------------------------------------
# UI INPUT
# --------------------------------------------------
user_query = st.text_input(
    "💬 Ask a support question",
    placeholder="Why is my SeeQ connector failing in Synapse?"
)

# --------------------------------------------------
# RUN RAG
# --------------------------------------------------
if user_query:
    with st.spinner("🔍 Searching past tickets & generating answer..."):
        response = rag_chain.invoke({"input": user_query})

    retrieved_docs = response["context"]
    answer = response["answer"]

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------
    col1, col2 = st.columns([1, 1.3])

    # ---------------- LEFT: TICKETS ----------------
    with col1:
        st.subheader("📌 Top 3 Relevant Tickets")

        for idx, doc in enumerate(retrieved_docs, start=1):
            ticket_id = doc.metadata.get("ticket_id", "UNKNOWN")
            content_preview = doc.page_content[:400]

            with st.expander(f"{idx}. Ticket ID: {ticket_id}", expanded=False):
                st.write(content_preview)

    # ---------------- RIGHT: ANSWER ----------------
    with col2:
        st.subheader("🤖 Agent Answer")
        st.markdown(answer)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("🔐 Powered by Gemini + Pinecone | L3 Support RAG Agent")
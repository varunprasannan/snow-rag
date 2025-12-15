import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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
INCIDENT_CSV_PATH = "new_tickets.csv"

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
# CACHED LOADERS
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

@st.cache_data
def load_incidents():
    return pd.read_csv(INCIDENT_CSV_PATH)

# --------------------------------------------------
# LOAD COMPONENTS
# --------------------------------------------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = load_llm()
incidents_df = load_incidents()

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
system_prompt = (
    "You are an expert L3 Support Engineer for Azure Synapse, "
    "Azure Data Factory, and SeeQ.\n\n"
    "Use the following resolved support tickets to answer the incident.\n"
    "Always cite the ticket IDs you used in a section called 'Sources'.\n"
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
# UI: INCIDENT NUMBER INPUT
# --------------------------------------------------
incident_number = st.text_input(
    "🆘 Enter Incident Number",
    placeholder="INC051"
)

# --------------------------------------------------
# PROCESS INCIDENT
# --------------------------------------------------
if incident_number:
    incident_row = incidents_df[
        incidents_df["incident_number"] == incident_number
    ]

    if incident_row.empty:
        st.error("❌ Incident number not found.")
    else:
        service = incident_row.iloc[0]["Service"]
        description = incident_row.iloc[0]["description"]

        # Auto-generated query
        generated_query = (
            f"Service: {service}\n"
            f"Incident Description: {description}"
        )

        # Show incident details
        st.info("📄 Incident Details Retrieved")
        st.markdown(f"**Service:** {service}")
        st.markdown(f"**Description:** {description}")

        with st.spinner("🔍 Finding similar resolved tickets..."):
            response = rag_chain.invoke(
                {"input": generated_query}
            )

        retrieved_docs = response["context"]
        answer = response["answer"]

        # --------------------------------------------------
        # RESULTS LAYOUT
        # --------------------------------------------------
        col1, col2 = st.columns([1, 1.3])

        # ---------------- LEFT: TICKETS ----------------
        with col1:
            st.subheader("📌 Top 3 Relevant Past Tickets")

            for idx, doc in enumerate(retrieved_docs, start=1):
                ticket_id = doc.metadata.get("ticket_id", "UNKNOWN")
                preview = doc.page_content[:400]

                with st.expander(f"{idx}. Ticket ID: {ticket_id}"):
                    st.write(preview)

        # ---------------- RIGHT: ANSWER ----------------
        with col2:
            st.subheader("🤖 Suggested Resolution")
            st.markdown(answer)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("🔐 Gemini + Pinecone | L3 Incident Resolution Assistant")

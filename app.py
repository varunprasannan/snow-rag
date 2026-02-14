# Streamlit app: L3 Support AI Agent using Azure OpenAI + Azure AI Search
# Incremental wiring: intent-based routing + notebook retrieval (safe, non-breaking)

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import Any, List

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ConfigDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# NEW IMPORTS
from rag.intent_router import detect_intent
from rag.intent import QueryIntent
from retrievers.notebook_retriever import AzureNotebookRetriever

# --------------------------------------------------
# CSV LOADERS
# --------------------------------------------------
@st.cache_data
def load_support_tickets():
    return pd.read_csv("support_tickets.csv")

@st.cache_data
def load_new_tickets():
    return pd.read_csv("new_tickets.csv")

try:
    support_tickets_df = load_support_tickets()
    new_tickets_df = load_new_tickets()
except Exception:
    st.error("Failed to load CSV files")
    st.stop()

# --------------------------------------------------
# ENV & CONFIG
# --------------------------------------------------
load_dotenv()

TICKET_INDEX = "support-agent-v2-azure-embeddings"
NOTEBOOK_INDEX = "engineering-notebooks-index"
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

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    options=[
        "🤖 AI Support Agent",
        "📂 Resolved Tickets",
        "🆕 New Incidents"
    ]
)

# ==================================================
# MAIN APP
# ==================================================
if page == "🤖 AI Support Agent":
    st.title("🤖 L3 Support AI Agent")
    st.caption("Azure Synapse • Azure Data Factory • SeeQ")

    # --------------------------------------------------
    # CACHED LOADERS
    # --------------------------------------------------
    @st.cache_resource
    def load_embeddings():
        return AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            api_version="2024-02-15-preview"
        )

    @st.cache_resource
    def load_llm():
        return AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0
        )

    @st.cache_resource
    def load_ticket_search_client():
        return SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=TICKET_INDEX,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )

    @st.cache_resource
    def load_notebook_search_client():
        return SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=NOTEBOOK_INDEX,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )

    @st.cache_data
    def load_incidents():
        return pd.read_csv(INCIDENT_CSV_PATH)

    # --------------------------------------------------
    # TICKET RETRIEVER
    # --------------------------------------------------
    class AzureSearchRetriever(BaseRetriever):
        search_client: Any = Field(...)
        embeddings: Any = Field(...)
        k: int = Field(default=3)

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            query_vector = self.embeddings.embed_query(query)

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=self.k,
                fields="embedding"
            )

            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                select=["ticket_id", "content"]
            )

            return [
                Document(
                    page_content=r["content"],
                    metadata={"ticket_id": r["ticket_id"]}
                )
                for r in results
            ]

    # --------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------
    embeddings = load_embeddings()
    llm = load_llm()

    ticket_retriever = AzureSearchRetriever(
        search_client=load_ticket_search_client(),
        embeddings=embeddings,
        k=TOP_K
    )

    notebook_retriever = AzureNotebookRetriever(
        search_client=load_notebook_search_client(),
        embeddings=embeddings,
        k=TOP_K
    )

    incidents_df = load_incidents()

    # --------------------------------------------------
    # QUERY MODE
    # --------------------------------------------------
    query_mode = st.radio(
        "🔎 How would you like to search?",
        ["Natural Language Query", "Incident Number"],
        horizontal=True
    )

    user_input = st.text_input(
        "💬 Enter your query",
        placeholder=(
            "Describe the issue you're facing..."
            if query_mode == "Natural Language Query"
            else "Enter Incident Number (e.g. INC051)"
        )
    )

    # --------------------------------------------------
    # PROCESS QUERY
    # --------------------------------------------------
    if user_input:
        if query_mode == "Incident Number":
            incident_row = incidents_df[
                incidents_df["incident_number"] == user_input
            ]

            if incident_row.empty:
                st.error("❌ Incident number not found.")
                st.stop()

            service = incident_row.iloc[0]["Service"]
            description = incident_row.iloc[0]["description"]

            final_query = f"Service: {service}\nIncident Description: {description}"

            st.info("📄 Incident Details Retrieved")
            st.markdown(f"**Service:** {service}")
            st.markdown(f"**Description:** {description}")

        else:
            final_query = user_input

        intent = detect_intent(final_query, llm)

        ticket_docs = []
        notebook_docs = []

        if intent in {
            QueryIntent.ACCESS_ISSUE,
            QueryIntent.SCHEMA_CHANGE,
            QueryIntent.GENERAL
        }:
            ticket_docs = ticket_retriever.invoke(final_query)

        if intent in {
            QueryIntent.SCHEMA_CHANGE,
            QueryIntent.PIPELINE_LOGIC
        }:
            notebook_docs = notebook_retriever.invoke(final_query)

        # --------------------------------------------------
        # BUILD CONTEXT
        # --------------------------------------------------
        context_blocks = []

        if ticket_docs:
            context_blocks.append("### Resolved Support Tickets")
            for d in ticket_docs:
                context_blocks.append(
                    f"- Ticket {d.metadata.get('ticket_id')}:\n{d.page_content}"
                )

        if notebook_docs:
            context_blocks.append("\n### Engineering Notebooks")
            for d in notebook_docs:
                context_blocks.append(
                    f"- Notebook {d.metadata['notebook_name']} "
                    f"(Cell {d.metadata['cell_index']}):\n{d.page_content}"
                )

        context = "\n\n".join(context_blocks)

        # --------------------------------------------------
        # LLM CALL
        # --------------------------------------------------
        system_prompt = f"""
You are an expert L3 Support Engineer and Data Platform Engineer.

Use the provided context to answer the user's request.
Always list sources at the end. Please remember, if you are working with notebooks by any chance, do not assume any joins will be made by the user. Rather mention how to make the join too.

Context:
{context}
"""

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_query}
        ])

        # --------------------------------------------------
        # UI OUTPUT
        # --------------------------------------------------
        col1, col2 = st.columns([1, 1.3])

        with col1:
            if ticket_docs:
                st.subheader("📌 Relevant Support Tickets")
                for d in ticket_docs:
                    with st.expander(d.metadata.get("ticket_id")):
                        st.write(d.page_content)

            if notebook_docs:
                st.subheader("📓 Relevant Engineering Notebooks")
                for d in notebook_docs:
                    with st.expander(
                        f"{d.metadata['notebook_name']} "
                        f"(Cell {d.metadata['cell_index']})"
                    ):
                        st.code(d.page_content, language="python")

        with col2:
            st.subheader("🤖 Suggested Resolution")
            st.markdown(response.content)

# ==================================================
# OTHER PAGES
# ==================================================
elif page == "📂 Resolved Tickets":
    st.title("📂 Resolved Support Tickets")
    st.dataframe(support_tickets_df, use_container_width=True, height=700)

elif page == "🆕 New Incidents":
    st.title("🆕 Incoming Incidents")
    st.dataframe(new_tickets_df, use_container_width=True, height=700)

st.markdown("---")
st.caption("🔐 Azure OpenAI + Azure AI Search | L3 Incident Resolution Assistant")

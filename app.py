# Streamlit app: L3 Support AI Agent using Azure OpenAI + Azure AI Search

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import Any, List


from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ConfigDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

@st.cache_data
def load_support_tickets():
    return pd.read_csv("support_tickets.csv")

@st.cache_data
def load_new_tickets():
    return pd.read_csv("new_tickets.csv")

try:
    support_tickets_df = load_support_tickets()
    new_tickets_df = load_new_tickets()
except Exception as e:
    st.error("Failed to load support_tickets.csv")
    st.stop()

# --------------------------------------------------
# ENV & CONFIG
# --------------------------------------------------
load_dotenv()

INDEX_NAME = "support-agent-v2-azure-embeddings"
TOP_K = 3
INCIDENT_CSV_PATH = "new_tickets.csv"

# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    options=[
        "🤖 AI Support Agent",
        "📂 Resolved Tickets",
        "🆕 New Incidents"
    ]
)

st.set_page_config(
    page_title="L3 Support AI Agent",
    page_icon="🤖",
    layout="wide"
)

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
    def load_search_client():
        return SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )

    @st.cache_data
    def load_incidents():
        return pd.read_csv(INCIDENT_CSV_PATH)

    # --------------------------------------------------
    # AZURE SEARCH RETRIEVER (LangChain-compatible)
    # --------------------------------------------------
    class AzureSearchRetriever(BaseRetriever):
        search_client: Any = Field(...)
        embeddings: Any = Field(...)
        k: int = Field(default=3)

        model_config = ConfigDict(
            arbitrary_types_allowed=True
        )

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

            docs = []
            for r in results:
                docs.append(
                    Document(
                        page_content=r["content"],
                        metadata={"ticket_id": r["ticket_id"]}
                    )
                )

            return docs

    # --------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------
    embeddings = load_embeddings()
    llm = load_llm()
    search_client = load_search_client()
    incidents_df = load_incidents()

    retriever = AzureSearchRetriever(
        search_client=search_client,
        embeddings=embeddings,
        k=TOP_K
    )

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
    # QUERY MODE SELECTION
    # --------------------------------------------------
    query_mode = st.radio(
        "🔎 How would you like to search?",
        ["Natural Language Query", "Incident Number"],
        horizontal=True
    )

    # --------------------------------------------------
    # INPUT BOX (shared)
    # --------------------------------------------------
    user_input = st.text_input(
        "💬 Enter your query",
        placeholder=(
            "Describe the issue you're facing..."
            if query_mode == "Natural Language Query"
            else "Enter Incident Number (e.g. INC051)"
        )
    )

    # --------------------------------------------------
    # PROCESS INCIDENT
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

            generated_query = (
                f"Service: {service}\n"
                f"Incident Description: {description}"
            )

            st.info("📄 Incident Details Retrieved")
            st.markdown(f"**Service:** {service}")
            st.markdown(f"**Description:** {description}")

            final_query = generated_query

        else:
            # Natural language query mode
            st.info("💬 Processing natural language query")
            final_query = user_input

        with st.spinner("🔍 Finding similar resolved tickets..."):
            response = rag_chain.invoke({"input": final_query})

        retrieved_docs = response["context"]
        answer = response["answer"]

        # --------------------------------------------------
        # RESULTS LAYOUT
        # --------------------------------------------------
        col1, col2 = st.columns([1, 1.3])

        # ---------------- LEFT: TICKETS ----------------
        with col1:
            st.subheader("📌 Top Relevant Past Tickets")

            for idx, doc in enumerate(retrieved_docs, start=1):
                ticket_id = doc.metadata.get("ticket_id", "UNKNOWN")
                preview = doc.page_content[:400]

                with st.expander(f"{idx}. Ticket ID: {ticket_id}"):
                    st.write(preview)

        # ---------------- RIGHT: ANSWER ----------------
        with col2:
            st.subheader("🤖 Suggested Resolution")
            st.markdown(answer)

elif page == "📂 Resolved Tickets":
    st.title("📂 Resolved Support Tickets")
    st.caption("Historical tickets used by the AI agent")

    st.dataframe(
        support_tickets_df,
        use_container_width=True,
        height=700
    )

elif page == "🆕 New Incidents":
    st.title("🆕 Incoming Incidents")
    st.caption("Live incidents from ServiceNow / CSV")

    st.dataframe(
        new_tickets_df,
        use_container_width=True,
        height=700
    )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------
st.markdown("---")
st.caption("🔐 Azure OpenAI + Azure AI Search | L3 Incident Resolution Assistant")


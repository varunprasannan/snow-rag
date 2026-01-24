# This app has the ingestion logic and chat in the terminal using Azure OpenAI and Azure AI Search

import os
import time
from dotenv import load_dotenv
from typing import Any, List
from pydantic import Field, ConfigDict

# --- LangChain Imports ---
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# --- Azure Search Imports ---
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")
CSV_FILE_PATH = "support_tickets.csv"

# --- Azure OpenAI Embeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    api_version="2024-02-15-preview"
)

# --- Azure Search Clients ---
def get_index_client():
    return SearchIndexClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
    )

def get_search_client():
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
    )

# ======================================================
#              CREATE AZURE SEARCH INDEX
# ======================================================


def create_search_index():
    index_client = get_index_client()

    if INDEX_NAME in index_client.list_index_names():
        print(f"✔ Index '{INDEX_NAME}' already exists.")
        return

    print(f"📦 Creating Azure Search index '{INDEX_NAME}'...")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="ticket_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algorithm"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algorithm"
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_index(index)
    print("✅ Azure Search index created successfully")

# ======================================================
#              INGEST CSV → AZURE SEARCH
# ======================================================
def setup_vector_db():
    """
    INGEST MODE: Reads CSV, chunks it (1 row = 1 chunk), and uploads to Azure AI Search.
    """
    print("🚀 Connecting to Azure AI Search...")
    create_search_index()

    loader = CSVLoader(file_path=CSV_FILE_PATH, encoding="utf-8")
    docs = loader.load()

    print(f"🧩 Loaded {len(docs)} tickets. Uploading to Azure Search...")

    search_client = get_search_client()
    batch = []

    for i, doc in enumerate(docs):
        content = doc.page_content
        ticket_id = doc.metadata.get("ticket_id", f"TICKET_{i}")

        vector = embeddings.embed_documents([content])[0]

        batch.append({
            "id": str(i),
            "ticket_id": ticket_id,
            "content": content,
            "embedding": vector
        })

    search_client.upload_documents(batch)
    print("✅ Ingestion Complete! You can now run the chat mode.")

# ======================================================
#           CUSTOM AZURE SEARCH RETRIEVER
# ======================================================

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
            search_text="",   # MUST be empty string, not None
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

# ======================================================
#                   CHAT AGENT
# ======================================================
def start_chat_agent():
    """
    CHAT MODE: RAG Pipeline using Azure OpenAI and Azure AI Search.
    """
    print("🤖 Initializing AI Agent...")

    retriever = AzureSearchRetriever(
        search_client=get_search_client(),
        embeddings=embeddings,
        k=3
    )

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-02-15-preview",
        temperature=0
    )


    system_prompt = (
        "You are an expert L3 Support Engineer for Azure Synapse, Azure Data Factory, and SeeQ."
        "\n\n"
        "Use the following resolved support tickets to answer the user's question."
        " At the end of your answer, include a section called 'Sources' and list the ticket IDs used."
        " If the answer is not present in the tickets, say that you don't know."
        "\n\n"
        "Resolved Tickets:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n💬 Support Agent Ready! (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke({"input": user_input})

        print("\n📌 Top 3 Relevant Past Tickets:")
        print("-" * 50)

        for idx, doc in enumerate(response["context"], start=1):
            ticket_id = doc.metadata.get("ticket_id", "UNKNOWN")
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"{idx}. Ticket ID: {ticket_id}")
            print(f"   Summary: {preview}...")
            print("-" * 50)

        print("\n🤖 Agent Answer:")
        print(response["answer"])
        print("-" * 50)

# ======================================================
#                      MAIN
# ======================================================
if __name__ == "__main__":
    choice = input("Do you want to (1) Ingest CSV Data or (2) Chat? [Enter 1 or 2]: ")

    if choice == "1":
        setup_vector_db()
    elif choice == "2":
        start_chat_agent()
    else:
        print("Invalid choice.")

import os
import re
import nbformat
from dotenv import load_dotenv
from typing import List

from azure.core.credentials import AzureKeyCredential
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

from langchain_openai import AzureOpenAIEmbeddings

def make_safe_id(text: str) -> str:
    """
    Azure Search document keys may only contain
    letters, digits, underscore, dash, or equals.
    """
    return re.sub(r"[^a-zA-Z0-9_\-=]", "_", text)

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

NOTEBOOK_INDEX_NAME = "engineering-notebooks-index"
NOTEBOOK_DIR = "./notebooks"  # folder containing .ipynb files

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    api_version="2024-02-15-preview"
)

# --------------------------------------------------
# AZURE SEARCH CLIENTS
# --------------------------------------------------
def get_index_client():
    return SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

def get_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=NOTEBOOK_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

# --------------------------------------------------
# CREATE NOTEBOOK INDEX
# --------------------------------------------------
def create_notebook_index():
    index_client = get_index_client()

    if NOTEBOOK_INDEX_NAME in index_client.list_index_names():
        print(f"✔ Index '{NOTEBOOK_INDEX_NAME}' already exists.")
        return

    print(f"📦 Creating notebook index '{NOTEBOOK_INDEX_NAME}'...")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),

        SearchField(name="content", type=SearchFieldDataType.String),

        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile"
        ),

        SearchField(name="notebook_name", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="domain", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="table_name", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="operation_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="cell_index", type=SearchFieldDataType.Int32)
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-algorithm")
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algorithm"
            )
        ]
    )

    index = SearchIndex(
        name=NOTEBOOK_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_index(index)
    print("✅ Notebook index created")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def classify_operation(cell_text: str) -> str:
    text = cell_text.lower()

    if "create table" in text:
        return "create_table"
    if "insert overwrite" in text or ".write.mode" in text:
        return "insert"
    if "vacuum" in text:
        return "vacuum"
    if "read.parquet" in text or "abfss://" in text:
        return "load"
    return "other"

def parse_notebook_metadata(notebook_name: str):
    # Example: SC_factPurchaseRequisition.ipynb
    base = notebook_name.replace(".ipynb", "")
    parts = base.split("_", 1)

    domain = parts[0]
    table_name = parts[1] if len(parts) > 1 else "unknown"

    return domain, table_name

# --------------------------------------------------
# INGEST NOTEBOOKS
# --------------------------------------------------
def ingest_notebooks():
    create_notebook_index()
    search_client = get_search_client()

    documents = []

    for file in os.listdir(NOTEBOOK_DIR):
        if not file.endswith(".ipynb"):
            continue

        notebook_path = os.path.join(NOTEBOOK_DIR, file)
        print(f"📓 Processing {file}")

        nb = nbformat.read(notebook_path, as_version=4)
        domain, table_name = parse_notebook_metadata(file)

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type not in ["code", "markdown"]:
                continue

            content = cell.source.strip()
            if not content:
                continue

            operation_type = classify_operation(content)

            vector = embeddings.embed_query(content)

            raw_id = f"{file}_{idx}"
            doc_id = make_safe_id(raw_id)

            documents.append({
                "id": doc_id,
                "content": content,
                "embedding": vector,
                "notebook_name": file,
                "domain": domain,
                "table_name": table_name,
                "operation_type": operation_type,
                "cell_index": idx
            })

    if documents:
        search_client.upload_documents(documents)
        print(f"✅ Ingested {len(documents)} notebook cells")
    else:
        print("⚠ No notebook content found")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    ingest_notebooks()

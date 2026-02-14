from typing import Any, List
from pydantic import Field, ConfigDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient

class AzureNotebookRetriever(BaseRetriever):
    """
    Retrieves relevant engineering notebook cells from Azure AI Search
    """
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
            select=[
                "content",
                "notebook_name",
                "domain",
                "table_name",
                "operation_type",
                "cell_index"
            ]
        )

        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["content"],
                    metadata={
                        "notebook_name": r.get("notebook_name"),
                        "domain": r.get("domain"),
                        "table_name": r.get("table_name"),
                        "operation_type": r.get("operation_type"),
                        "cell_index": r.get("cell_index")
                    }
                )
            )

        return docs

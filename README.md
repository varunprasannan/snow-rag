# L3 Support RAG Agent

A small collection of scripts and Streamlit apps that demonstrate a Retrieval-Augmented Generation (RAG) workflow for L3 support incidents using Gemini / Azure OpenAI and Pinecone as the vector store. The project ingests resolved support tickets (CSV), indexes them in Pinecone, then answers new incidents by retrieving similar past tickets and composing an answer.

**What it uses**
- **LangChain**-style chains and connectors (Google/Azure LLM + embeddings)
- **Pinecone** for vector storage and retrieval
- **Streamlit** for two lightweight web UIs
- Simple CSVs as the knowledge base
- `python-dotenv` for environment configuration

**Repository Files**
- [app.py](app.py): Streamlit RAG UI to directly query the knowledge base and show sources.
- [st_snow_gemini.py](st_snow_gemini.py): Streamlit incident resolver that reads `new_tickets.csv`, retrieves similar tickets and suggests a resolution using Gemini.
- [agent_gemini.py](agent_gemini.py): Terminal agent using Gemini embeddings/LLM + Pinecone. Supports ingestion and interactive chat.
- [agent_openai.py](agent_openai.py): Terminal agent using Azure OpenAI embeddings/LLM + Pinecone. Supports ingestion and interactive chat.
- [support_tickets.csv](support_tickets.csv): The primary KB — resolved tickets used to build the index.
- [new_tickets.csv](new_tickets.csv): Example incoming incidents (used by the Streamlit incident resolver).

Quick notes: several files include a short comment at the top explaining their purpose.

Prerequisites
- Python 3.10+ recommended
- A Pinecone account and API key
- Google or Azure credentials depending on which agent you run

Environment variables
Create a `.env` file in the project root with the keys your chosen connector needs. Typical variables used by the included scripts are:

- `PINECONE_API_KEY` — Pinecone API key
- Google/Gemini (if using Gemini files): set whichever auth the Google SDK / langchain_google_genai library expects (e.g. `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS`)
- Azure (if using Azure files): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and any deployment names required by your Azure setup

Install (example)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip show streamlit pandas python-dotenv pinecone langchain-google-genai
```

Usage

1) Ingest CSV to Pinecone (creates the index and uploads ticket embeddings)

For Gemini-based ingestion:
```bash
python agent_gemini.py
# Choose option 1 when prompted (Ingest CSV Data)
```

For Azure OpenAI-based ingestion:
```bash
python agent_openai.py
# Choose option 1 when prompted (Ingest CSV Data)
```

2) Chat (terminal RAG agent)

After ingestion you can run the chat mode to ask questions using the indexed tickets:
```bash
python agent_gemini.py
# or
python agent_openai.py
```

3) Streamlit UIs

Ask the knowledge base directly (general query):
```bash
streamlit run app.py
```

Resolve an incoming incident from `new_tickets.csv` (enter an incident number):
```bash
streamlit run st_snow_gemini.py
```

Files of interest
- [support_tickets.csv](support_tickets.csv) — resolved tickets used to build the KB.
- [new_tickets.csv](new_tickets.csv) — sample incoming incidents used by the Streamlit incident resolver.

Troubleshooting
- If index creation fails, check your `PINECONE_API_KEY` and Pinecone account/region limits.
- If embeddings/LLM calls fail, verify relevant cloud credentials and deployment names for Google/Azure.
- If retrieval returns poor matches, try increasing `k` (the number of retrieved docs) or re-tune embedding model parameters.

- Next steps & suggestions
- Keep `requirements.txt` updated after dependency changes.
- Add a `pyproject.toml` for modern packaging and tooling if desired.
- Add a small ingestion log / metadata writer so you can re-run ingestion idempotently.
- Add unit tests for the CSV loader and a small smoke test for the RAG chain.

License & Author
- Lightweight demo for internal use. Author: Varun Prasannan
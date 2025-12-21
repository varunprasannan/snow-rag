# This app has the ingestion logic and chat in the terminal using Gemini and Pinecone

import os
import time
from dotenv import load_dotenv

# --- Latest LangChain Imports (Dec 2025 Standard) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# Load API Keys from .env file
load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = "support-agent-v1"
CSV_FILE_PATH = "support_tickets.csv"  # Ensure this file exists

# 1. Initialize Gemini Embeddings (Free tier uses 'models/text-embedding-004')
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Initialize Pinecone Client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def setup_vector_db():
    """
    INGEST MODE: Reads CSV, chunks it (1 row = 1 chunk), and uploads to Pinecone.
    """
    print(f"🚀 Connecting to Pinecone...")
    
    # Check if index exists, if not create it (Serverless - AWS us-east-1 is usually free tier compatible)
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"📦 Creating new index '{INDEX_NAME}'...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            print("Tip: If you are on the legacy Free Tier, create a 'Starter' index manually in the UI.")
            return

    print("📄 Loading CSV data...")
    # CSVLoader automatically creates 1 Document per Row (Perfect for tickets)
    loader = CSVLoader(file_path=CSV_FILE_PATH, encoding="utf-8")
    docs = loader.load()
    
    print(f"🧩 Loaded {len(docs)} tickets. Uploading to Pinecone (this may take a moment)...")
    
    # Upsert to Pinecone
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("✅ Ingestion Complete! You can now run the chat mode.")

def start_chat_agent():
    """
    CHAT MODE: RAG Pipeline using Gemini Pro and Pinecone.
    """
    print("🤖 Initializing AI Agent...")
    
    # 1. Connect to existing Index
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 similar tickets

    # 2. Initialize Gemini Pro (LLM)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # 3. Create the Prompt Template
    system_prompt = (
    "You are an expert L3 Support Engineer for Azure Synapse, Azure Data Factory, and SeeQ."
    "\n\n"
    "Use the following resolved support tickets to answer the user's question."
    "At the end of your answer, include a section called 'Sources' and list the ticket IDs used."
    "If the answer is not present in the tickets, say that you don't know."
    "\n\n"
    "Resolved Tickets:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4. Build the Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n💬 Support Agent Ready! (Type 'exit' to quit)")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Run the RAG pipeline
        response = rag_chain.invoke({"input": user_input})

        # --- Display Retrieved Tickets ---
        print("\n📌 Top 3 Relevant Past Tickets:")
        print("-" * 50)

        for idx, doc in enumerate(response["context"], start=1):
            ticket_id = doc.metadata.get("ticket_id", "UNKNOWN")
            preview = doc.page_content[:300].replace("\n", " ")
            
            print(f"{idx}. Ticket ID: {ticket_id}")
            print(f"   Summary: {preview}...")
            print("-" * 50)

        # --- Display Final Answer ---
        print("\n🤖 Agent Answer:")
        print(response["answer"])
        print("-" * 50)


# --- MAIN MENU ---
if __name__ == "__main__":
    choice = input("Do you want to (1) Ingest CSV Data or (2) Chat? [Enter 1 or 2]: ")
    
    if choice == "1":
        setup_vector_db()
    elif choice == "2":
        start_chat_agent()
    else:
        print("Invalid choice.")
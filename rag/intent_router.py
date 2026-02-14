import re
from rag.intent import QueryIntent

SCHEMA_KEYWORDS = [
    "add column", "add field", "add attribute",
    "modify table", "alter table", "schema change",
    "add maktl", "add lifnr", "add aedat"
]

PIPELINE_KEYWORDS = [
    "notebook", "etl", "pipeline",
    "insert overwrite", "create table",
    "delta", "spark", "sql"
]

ACCESS_KEYWORDS = [
    "cannot access", "not able to access",
    "login issue", "permission denied",
    "seeq error", "access issue"
]


def rule_based_intent(query: str) -> QueryIntent | None:
    q = query.lower()

    if any(k in q for k in ACCESS_KEYWORDS):
        return QueryIntent.ACCESS_ISSUE

    if any(k in q for k in SCHEMA_KEYWORDS):
        return QueryIntent.SCHEMA_CHANGE

    if any(k in q for k in PIPELINE_KEYWORDS):
        return QueryIntent.PIPELINE_LOGIC

    return None

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rag.intent import QueryIntent

def llm_intent_classifier(query: str, llm: AzureChatOpenAI) -> QueryIntent:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the user's query into ONE of the following intents:\n"
         "- access_issue\n"
         "- schema_change\n"
         "- pipeline_logic\n"
         "- general\n\n"
         "Return ONLY the intent name."),
        ("human", "{query}")
    ])

    response = llm.invoke(prompt.format(query=query))
    intent_str = response.content.strip().lower()

    try:
        return QueryIntent(intent_str)
    except ValueError:
        return QueryIntent.GENERAL
    
def detect_intent(query: str, llm: AzureChatOpenAI) -> QueryIntent:
    intent = rule_based_intent(query)
    if intent:
        return intent

    return llm_intent_classifier(query, llm)


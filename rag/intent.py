from enum import Enum

class QueryIntent(str, Enum):
    ACCESS_ISSUE = "access_issue"
    SCHEMA_CHANGE = "schema_change"
    PIPELINE_LOGIC = "pipeline_logic"
    GENERAL = "general"
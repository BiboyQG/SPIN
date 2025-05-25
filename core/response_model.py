from pydantic import BaseModel
from typing import List

class ResponseOfReflectionResearchState(BaseModel):
    found_info: str
    missing_info: str
    difficulties: str
    strategies: str
    gap_summary: str

class ResponseOfReflectionSubQuestions(BaseModel):
    questions: List[str]

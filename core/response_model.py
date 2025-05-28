from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class ResponseOfReflectionResearchState(BaseModel):
    found_info: str
    missing_info: str
    difficulties: str
    strategies: str
    gap_summary: str


class ResponseOfReflectionSubQuestions(BaseModel):
    questions: List[str]


class Action(str, Enum):
    visit = "visit"
    search = "search"
    reflect = "reflect"
    evaluate = "evaluate"


class ResponseOfActionPlan(BaseModel):
    action: Action
    reason: str


class ResponseOfConsolidation(BaseModel):
    value: str
    sources: List[str]
    reasoning: str


class ResponseOfSelection(BaseModel):
    selected_urls: Optional[List[str]]
    reasoning: str


class ResponseOfWorthVisiting(BaseModel):
    worth_visiting: bool
    reason: str

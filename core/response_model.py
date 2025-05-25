from pydantic import BaseModel
from typing import List
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

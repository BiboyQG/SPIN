from typing import List, Optional, Union
from pydantic import BaseModel, Field


class Instructor(BaseModel):
    name: str
    email: str
    office_hours: Optional[List[str]] = None
    office_location: Optional[str] = None


class TextBook(BaseModel):
    title: str
    authors: List[str]
    link: Optional[str] = None


class GradingComponent(BaseModel):
    name: str
    weight: float = Field(description="E.g. 0.05 for 5%")
    description: Optional[str] = None


class Schedule(BaseModel):
    day: str
    time_start: str
    time_end: str
    location: str


class PreRequisite(BaseModel):
    course_code: str
    minimum_grade: Optional[str]
    can_be_concurrent: bool


class TeachingAssistant(BaseModel):
    name: str
    email: str
    office_hours: Optional[List[str]] = None
    office_location: Optional[str] = None


class Course(BaseModel):
    code: str
    title: str
    credits: Union[int, float]
    description: str
    semester: str
    year: int
    instructors: List[Instructor]
    teaching_assistants: Optional[List[TeachingAssistant]] = None
    schedule: List[Schedule]
    prerequisites: Optional[List[PreRequisite]] = None
    textbooks: Optional[List[TextBook]] = None
    grading_components: List[GradingComponent]
    syllabus_link: Optional[str] = None
    department: str
    learning_objectives: List[str]

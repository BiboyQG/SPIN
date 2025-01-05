from typing import List, Optional, Union
from pydantic import BaseModel, Field
import json


class Instructor(BaseModel):
    name: str
    email: str
    office_hours: Optional[List[str]] = Field(None, alias="officeHours")
    office_location: Optional[str] = Field(None, alias="officeLocation")


class TextBook(BaseModel):
    title: str
    authors: List[str]
    isbn: Optional[str] = None
    required: bool = True
    edition: Optional[str] = None
    publisher: Optional[str] = None


class GradingComponent(BaseModel):
    name: str
    weight: float
    description: Optional[str] = None


class Schedule(BaseModel):
    day: str
    time_start: str = Field(..., alias="timeStart")
    time_end: str = Field(..., alias="timeEnd")
    location: str


class PreRequisite(BaseModel):
    course_code: str = Field(..., alias="courseCode")
    minimum_grade: Optional[str] = Field(None, alias="minimumGrade")
    can_be_concurrent: bool = Field(False, alias="canBeConcurrent")


class TeachingAssistant(BaseModel):
    name: str
    email: str
    office_hours: Optional[List[str]] = Field(None, alias="officeHours")
    office_location: Optional[str] = Field(None, alias="officeLocation")


class Course(BaseModel):
    code: str
    title: str
    credits: Union[int, float]
    description: str
    semester: str
    year: int
    instructors: List[Instructor]
    teaching_assistants: Optional[List[TeachingAssistant]]
    schedule: List[Schedule]
    prerequisites: Optional[List[PreRequisite]] = None
    textbooks: Optional[List[TextBook]] = None
    grading_components: List[GradingComponent] = Field(..., alias="gradingComponents")
    syllabus_link: Optional[str] = Field(None, alias="syllabusLink")
    department: str
    learning_objectives: List[str] = Field(..., alias="learningObjectives")

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))

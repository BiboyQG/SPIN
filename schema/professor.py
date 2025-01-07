from pydantic import BaseModel, Field
from typing import List, Union
import json


class Education(BaseModel):
    degree: str
    field: str
    institution: str
    year: int


class ProfessionalHighlight(BaseModel):
    position: str
    organization: str
    year_start: int = Field(..., alias="yearStart")
    year_end: Union[int, None] = Field(..., alias="yearEnd")


class Publication(BaseModel):
    title: str
    authors: List[str]
    conference: str
    year: int


class TeachingHonor(BaseModel):
    honor: str
    year: int


class ResearchHonor(BaseModel):
    honor: str
    organization: str
    year: int


class Course(BaseModel):
    code: str
    title: str


class Contact(BaseModel):
    phone: str
    email: str


class Professor(BaseModel):
    fullname: str
    title: str = Field(
        description="The title of the professor, e.g. Assistant Professor, Teaching Professor, Gies RC Evans Innovation Fellow, etc."
    )
    contact: Contact
    office: str
    education: List[Education]
    biography: str
    professional_highlights: List[ProfessionalHighlight] = Field(
        ..., alias="professionalHighlights"
    )
    research_statement: str = Field(..., alias="researchStatement")
    research_areas: List[str] = Field(..., alias="researchAreas")
    publications: List[Publication]
    teaching_honors: List[TeachingHonor] = Field(..., alias="teachingHonors")
    research_honors: List[ResearchHonor] = Field(..., alias="researchHonors")
    courses_taught: List[Course] = Field(..., alias="coursesTaught")

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))

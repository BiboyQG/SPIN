from pydantic import BaseModel, Field
from typing import List, Optional


class Education(BaseModel):
    degree: str
    field: str
    institution: str
    year: int


class ProfessionalHighlight(BaseModel):
    position: str
    organization: str
    year_start: int
    year_end: Optional[int] = None


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
    professional_highlights: List[ProfessionalHighlight]
    research_statement: str
    research_areas: List[str]
    publications: List[Publication]
    teaching_honors: List[TeachingHonor]
    research_honors: List[ResearchHonor]
    courses_taught: List[Course]

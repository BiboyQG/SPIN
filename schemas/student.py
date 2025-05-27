from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Education(BaseModel):
    institution: str
    degree: str
    major: str
    gpa: Optional[float] = None
    period: str = Field(
        description="The period of the education, e.g., 'Fall 2023 - Fall 2025'"
    )
    location: str
    core_modules: List[str]


class Publication(BaseModel):
    title: str
    authors: List[str]
    venue: str
    year: int
    status: Literal["submitted", "accepted"] = Field(
        description="The status of the publication, e.g., 'submitted', 'accepted'"
    )


class ResearchExperience(BaseModel):
    title: str
    institution: str
    advisor: str
    period: str = Field(
        description="The period of the research experience, e.g., 'Fall 2023 - Fall 2025'"
    )
    code_link: Optional[str] = None
    project_link: Optional[str] = None
    achievements: List[str]


class WorkExperience(BaseModel):
    position: str
    organization: str
    location: str
    period: str = Field(
        description="The period of the work experience, e.g., 'Fall 2023 - Fall 2025'"
    )
    responsibilities: List[str]


class Award(BaseModel):
    year: int
    title: str
    organization: str


class Skills(BaseModel):
    programming: List[str]
    tools: List[str]


class Contact(BaseModel):
    email: str
    phone: Optional[str] = None
    address: str


class Student(BaseModel):
    fullname: str
    contact: Contact
    education: List[Education]
    publications: List[Publication]
    research_experience: List[ResearchExperience]
    work_experience: List[WorkExperience]
    awards: List[Award]
    skills: Skills

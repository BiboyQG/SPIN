from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json


class Education(BaseModel):
    institution: str
    degree: str
    major: str
    gpa: Optional[float]
    period: str = Field(
        description="The period of the education, e.g., 'Fall 2023 - Fall 2025'"
    )
    location: str
    core_modules: List[str] = Field(..., alias="coreModules")


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
    code_link: Optional[str] = Field(..., alias="codeLink")
    project_link: Optional[str] = Field(..., alias="projectLink")
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
    phone: Optional[str]
    address: str
    portfolio: str


class Student(BaseModel):
    fullname: str
    contact: Contact
    education: List[Education]
    publications: List[Publication]
    research_experience: List[ResearchExperience] = Field(
        ..., alias="researchExperience"
    )
    work_experience: List[WorkExperience] = Field(..., alias="workExperience")
    awards: List[Award]
    skills: Skills

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))

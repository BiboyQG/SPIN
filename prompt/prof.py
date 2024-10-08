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


class ResearchInterest(BaseModel):
    area: str
    description: str


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
    name: str
    title: str
    contact: Contact
    office: str
    education: List[Education]
    biography: str
    professional_highlights: List[ProfessionalHighlight] = Field(..., alias="professionalHighlights")
    research_statement: str = Field(..., alias="researchStatement")
    research_interests: List[ResearchInterest] = Field(..., alias="researchInterests")
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

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_data",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["phone", "email"],
                },
                "office": {"type": "string"},
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "degree": {"type": "string"},
                            "field": {"type": "string"},
                            "institution": {"type": "string"},
                            "year": {"type": "integer"},
                        },
                        "required": ["degree", "field", "institution", "year"],
                    },
                },
                "biography": {"type": "string"},
                "professionalHighlights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "string"},
                            "organization": {"type": "string"},
                            "yearStart": {"type": "integer"},
                            "yearEnd": {"type": ["integer", "null"]},
                        },
                        "required": ["position", "organization", "yearStart"],
                    },
                },
                "researchStatement": {"type": "string"},
                "researchInterests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["area", "description"],
                    },
                },
                "researchAreas": {"type": "array", "items": {"type": "string"}},
                "publications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "conference": {"type": "string"},
                            "year": {"type": "integer"},
                        },
                        "required": ["title", "authors", "conference", "year"],
                    },
                },
                "teachingHonors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "honor": {"type": "string"},
                            "year": {"type": "integer"},
                        },
                        "required": ["honor", "year"],
                    },
                },
                "researchHonors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "honor": {"type": "string"},
                            "organization": {"type": "string"},
                            "year": {"type": "integer"},
                        },
                        "required": ["honor", "organization", "year"],
                    },
                },
                "coursesTaught": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["code", "title"],
                    },
                },
            },
            "required": [
                "name",
                "title",
                "contact",
                "office",
                "education",
                "biography",
                "professionalHighlights",
                "researchStatement",
                "researchInterests",
                "researchAreas",
                "publications",
                "teachingHonors",
                "researchHonors",
                "coursesTaught",
            ],
            "additionalProperties": False,
        },
    },
}
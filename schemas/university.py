from pydantic import BaseModel, Field
from typing import List, Dict, Literal
import json


class Address(BaseModel):
    street: str
    city: str
    state: str
    postal_code: str
    country: str


class ContactInfo(BaseModel):
    phone: str
    email: str
    website: str

class Achievement(BaseModel):
    title: str
    year: int
    category: Literal["academic", "research", "sports"] = Field(description="e.g., academic, research, or sports")
    description: str


class University(BaseModel):
    name: str
    established_year: int = Field(..., alias="establishedYear")
    type: Literal["public", "private"] = Field(description="public/private")
    address: Address
    contact: ContactInfo
    president: str
    mission_statement: str = Field(..., alias="missionStatement")
    rankings: Dict[str, int] = Field(description="e.g., {'US News': 50, 'QS World': 100}")
    achievements: List[Achievement]
    international_students_percentage: float = Field(
        ..., alias="internationalStudentsPercentage"
    )

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))
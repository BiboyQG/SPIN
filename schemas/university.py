from pydantic import BaseModel, Field
from typing import List, Dict, Literal


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
    category: Literal["academic", "research", "sports"] = Field(
        description="e.g., academic, research, or sports"
    )
    description: str


class University(BaseModel):
    name: str
    established_year: int
    type: Literal["public", "private"] = Field(description="public/private")
    address: Address
    contact: ContactInfo
    president: str
    mission_statement: str
    rankings: Dict[str, int] = Field(
        description="e.g., {'US News': 50, 'QS World': 100}"
    )
    achievements: List[Achievement]
    international_students_percentage: float

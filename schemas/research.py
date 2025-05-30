from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from datetime import date


class Author(BaseModel):
    name: str
    institution: str
    email: Optional[str] = None


class Implementation(BaseModel):
    language: str
    repository_url: str
    documentation_url: Optional[str] = None
    requirements: List[str]


class Citation(BaseModel):
    title: str
    authors: List[str]
    venue: str
    year: int
    doi: Optional[str] = None
    citations_count: Optional[int] = None


class ResearchOutput(BaseModel):
    conference: str
    metrics: Dict[str, float] = Field(
        description="e.g., {'accuracy': 0.95, 'f1_score': 0.92}"
    )


class Dataset(BaseModel):
    name: str
    description: str


class Research(BaseModel):
    title: str
    authors: List[Author]
    abstract: str
    keywords: List[str]

    # Core research details
    problem_statement: str
    methodology: str
    contributions: List[str]

    # Technical details
    implementation: Implementation
    dataset: Optional[Dataset] = None  # Dataset details if applicable

    # Results and impact
    research_output: ResearchOutput

    # Publication and recognition
    primary_citation: Citation
    related_publications: List[Citation]

    # Additional metadata
    status: Literal["completed", "ongoing", "archived"] = Field(
        description="e.g., 'completed', 'ongoing', 'archived'"
    )
    start_date: date
    end_date: Optional[date] = None
    funding_sources: List[str]

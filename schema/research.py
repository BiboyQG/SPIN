from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from datetime import date
import json

class Author(BaseModel):
    name: str
    institution: str
    email: Optional[str] = None

class Implementation(BaseModel):
    language: str
    repository_url: str = Field(..., alias="repositoryUrl")
    documentation_url: Optional[str] = Field(None, alias="documentationUrl")
    requirements: List[str]

class Citation(BaseModel):
    title: str
    authors: List[str]
    venue: str
    year: int
    doi: Optional[str] = None
    citations_count: Optional[int] = Field(None, alias="citationsCount")

class ResearchOutput(BaseModel):
    conference: str
    metrics: Dict[str, float] = Field(description="e.g., {'accuracy': 0.95, 'f1_score': 0.92}")

class Dataset(BaseModel):
    name: str
    description: str

class Research(BaseModel):
    title: str
    authors: List[Author]
    abstract: str
    keywords: List[str]
    
    # Core research details
    problem_statement: str = Field(..., alias="problemStatement")
    methodology: str
    contributions: List[str]
    
    # Technical details
    implementation: Implementation
    dataset: Optional[Dataset]  # Dataset details if applicable
    
    # Results and impact
    research_output: ResearchOutput
    
    # Publication and recognition
    primary_citation: Citation = Field(..., alias="primaryCitation")
    related_publications: List[Citation] = Field(..., alias="relatedPublications")
    
    # Additional metadata
    status: Literal["completed", "ongoing", "archived"] = Field(description="e.g., 'completed', 'ongoing', 'archived'")
    start_date: date = Field(..., alias="startDate")
    end_date: Optional[date] = Field(None, alias="endDate")
    funding_sources: List[str] = Field(..., alias="fundingSources")
    
    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )
    
    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))
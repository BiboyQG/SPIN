from pydantic import BaseModel, Field
from typing import List, Optional

class UniversityFacts(BaseModel):
    year_founded: int = Field(..., alias="yearFounded")
    pulitzer_prizes: int = Field(..., alias="pulitzerPrizes")
    graphical_web_browser: str = Field(..., alias="graphicalWebBrowser")
    ranking: str = Field(..., alias="ranking")
    students: str = Field(..., alias="students")

class Mission(BaseModel):
    mission_statement: str = Field(..., alias="missionStatement")

class Vision(BaseModel):
    vision_statement: str = Field(..., alias="visionStatement")

class Faculty(BaseModel):
    notable_organizations: List[str] = Field(..., alias="notableOrganizations")
    awards: List[str] = Field(..., alias="awards")
    notable_alumni: List[str] = Field(..., alias="notableAlumni")

class AcademicResources(BaseModel):
    library_volumes: int = Field(..., alias="libraryVolumes")
    library_units: int = Field(..., alias="libraryUnits")
    online_catalog_visits: int = Field(..., alias="onlineCatalogVisits")
    computer_terminals: int = Field(..., alias="computerTerminals")

class Research(BaseModel):
    focus: str = Field(..., alias="focus")
    annual_funding: int = Field(..., alias="annualFunding")
    research_park: str = Field(..., alias="researchPark")
    nsf_ranking: str = Field(..., alias="nsfRanking")

class Arts(BaseModel):
    performing_arts_center: str = Field(..., alias="performingArtsCenter")
    museums: List[str] = Field(..., alias="museums")
    major_facilities: List[str] = Field(..., alias="majorFacilities")

class UndergraduateEducation(BaseModel):
    promise: str = Field(..., alias="promise")
    students: int = Field(..., alias="students")
    divisions: int = Field(..., alias="divisions")
    courses: int = Field(..., alias="courses")
    fields_of_study: int = Field(..., alias="fieldsOfStudy")
    degrees_awarded: int = Field(..., alias="degreesAwarded")

class Preeminence(BaseModel):
    excellence: str = Field(..., alias="excellence")
    impact: str = Field(..., alias="impact")
    leadership: str = Field(..., alias="leadership")
    diversity: str = Field(..., alias="diversity")

class University(BaseModel):
    name: str
    about: str
    facts: UniversityFacts
    mission: Mission
    vision: Vision
    campus_profile: str = Field(..., alias="campusProfile")
    faculty: Faculty
    academic_resources: AcademicResources
    research: Research
    arts: Arts
    undergraduate_education: UndergraduateEducation
    preeminence: Preeminence

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda field_name: "".join(
            word.capitalize() if i > 0 else word
            for i, word in enumerate(field_name.split("_"))
        )

    def json(self, **kwargs):
        return json.loads(super().json(by_alias=True, exclude_none=True, **kwargs))
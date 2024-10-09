from pydantic import BaseModel, Field
from typing import List
import json


class Person(BaseModel):
    name: str


class Star(Person):
    character: str


class Rating(BaseModel):
    value: float
    count: int


class TechnicalSpec(BaseModel):
    sound_mix: List[str] = Field(..., alias="soundMix")


class Popularity(BaseModel):
    rank: int
    change: int


class Movie(BaseModel):
    title: str
    year: int
    mpaa_rating: str = Field(..., alias="mpaaRating")
    runtime: str
    genres: List[str]
    plot: str
    directors: List[Person]
    writers: List[Person]
    stars: List[Star]
    imdb_rating: Rating = Field(..., alias="imdbRating")
    popularity: Popularity
    release_date: str
    country_of_origin: List[str] = Field(..., alias="countryOfOrigin")
    languages: List[str]
    filming_locations: List[str] = Field(..., alias="filmingLocations")
    production_companies: List[str] = Field(..., alias="productionCompanies")
    technical_specs: TechnicalSpec = Field(..., alias="technicalSpecs")

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
                "title": {"type": "string"},
                "year": {"type": "integer"},
                "mpaaRating": {"type": "string"},
                "runtime": {"type": "string"},
                "genres": {"type": "array", "items": {"type": "string"}},
                "plot": {"type": "string"},
                "directors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "writers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "stars": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "character": {"type": "string"},
                        },
                        "required": ["name", "character"],
                        "additionalProperties": False,
                    },
                },
                "imdbRating": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "number"},
                        "votes": {"type": "integer"},
                    },
                    "required": ["rating", "votes"],
                    "additionalProperties": False,
                },
                "popularity": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "change": {"type": "integer"},
                    },
                    "required": ["rank", "change"],
                    "additionalProperties": False,
                },
                "releaseDate": {"type": "string"},
                "countryOfOrigin": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "filmingLocations": {"type": "array", "items": {"type": "string"}},
                "productionCompanies": {"type": "array", "items": {"type": "string"}},
                "technicalSpecs": {
                    "type": "object",
                    "properties": {
                        "soundMix": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["soundMix"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "title",
                "year",
                "mpaaRating",
                "runtime",
                "genres",
                "plot",
                "directors",
                "writers",
                "stars",
                "imdbRating",
                "popularity",
                "releaseDate",
                "countryOfOrigin",
                "languages",
                "filmingLocations",
                "productionCompanies",
                "technicalSpecs",
            ],
            "additionalProperties": False,
        },
    },
}

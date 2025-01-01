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

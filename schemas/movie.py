from pydantic import BaseModel, Field
from typing import List


class Person(BaseModel):
    name: str


class Star(Person):
    character: str


class Rating(BaseModel):
    value: float
    count: int


class TechnicalSpec(BaseModel):
    sound_mix: List[str]


class Popularity(BaseModel):
    rank: int
    change: int


class Movie(BaseModel):
    title: str
    year: int
    mpaa_rating: str
    runtime: str
    genres: List[str]
    plot: str
    directors: List[Person]
    writers: List[Person]
    stars: List[Star]
    imdb_rating: Rating
    popularity: Popularity
    release_date: str
    country_of_origin: List[str]
    languages: List[str]
    filming_locations: List[str]
    production_companies: List[str]
    technical_specs: TechnicalSpec

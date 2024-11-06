from typing import Annotated, List
import json
import os

from pydantic import BaseModel, BeforeValidator, HttpUrl, TypeAdapter
from firecrawl import FirecrawlApp
from prompt.prof import Prof
from openai import OpenAI

fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

with open("./dataset/article/prof/0.txt", "r") as file:
    scrape_result = file.read()

open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"
prompt_type = "prof"

http_url_adapter = TypeAdapter(HttpUrl)

Url = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]


class RelatedLinks(BaseModel):
    related_links: List[Url]


def get_links_from_init_page(scrape_result):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format. Now you are given the initial page of the {prompt_type} article, please extract the most three related hyperlinks that may include information about the {prompt_type} from the page according to the context of the page and also the naming of the hyperlinks.",
            },
            {
                "role": "user",
                "content": f"The initial page of the {prompt_type} is: "
                + scrape_result,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": RelatedLinks.model_json_schema()},
    )
    return response.choices[0].message.content


def get_response_from_open_source_with_extra_body(scrape_result):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format.",
            },
            {
                "role": "user",
                "content": f"The article of the {prompt_type} is: " + scrape_result,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": Prof.model_json_schema()},
    )
    return response.choices[0].message.content


def get_response_from_open_source_with_extra_body_update(scrape_result, original_response):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format. Now you are given an inital {prompt_type} JSON structure, please update the JSON structure with the new information from the {prompt_type} article if necessary.",
            },
            {
                "role": "user",
                "content": f"The new {prompt_type} article is: " + scrape_result + "\n" + f"The initial {prompt_type} JSON structure is: " + original_response,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": Prof.model_json_schema()},
    )
    return response.choices[0].message.content

links_obj = RelatedLinks.model_validate_json(get_links_from_init_page(scrape_result))
links = links_obj.related_links

print(links)

# Two ways to get final information

# 1. Append the content of the links to the original page, and then extract the information based on the appended page
# 2. Extract the information from the original page, and update the information based on the extra links one by one

# The first way:

def get_final_information_from_all_links_in_one_time(scrape_result, links):
    for i, link in enumerate(links):
        scrape_result = (
            scrape_result
            + fire_app.scrape_url(
                link, params={"formats": ["markdown"], "excludeTags": ["img", "video"]}
            )["markdown"]
        )
        print(f"Finished updating context with link {i+1}")
    
    with open("test_all_links.json", "w") as f:
        json.dump(json.loads(get_response_from_open_source_with_extra_body(scrape_result)), f)
        print("Finished extracting information from all links in one time, saved to test_all_links.json")


## The second way:

def get_final_information_from_all_links_one_by_one(scrape_result, links):
   original_response = get_response_from_open_source_with_extra_body(scrape_result)
   for i, link in enumerate(links):
        original_response = get_response_from_open_source_with_extra_body_update(fire_app.scrape_url(link, params={"formats": ["markdown"], "excludeTags": ["img", "video"]})["markdown"], original_response)
        print(f"Finished updating JSON structure with link {i+1}")
   with open("test_one_by_one.json", "w") as f:
        json.dump(json.loads(original_response), f)
        print("Finished extracting information from all links one by one, saved to test_one_by_one.json")

# Compare the two ways

# get_final_information_from_all_links_in_one_time(scrape_result, links)
get_final_information_from_all_links_one_by_one(scrape_result, links)

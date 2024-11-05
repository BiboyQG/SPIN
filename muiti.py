from typing import Annotated, List
import os

# from firecrawl import FirecrawlApp
from pydantic import BaseModel, BeforeValidator, HttpUrl, TypeAdapter
from openai import OpenAI

# fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

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
                "content": f"The initial page of the {prompt_type} is: " + scrape_result,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={
            "guided_json": RelatedLinks.model_json_schema()
        }
    )
    return response.choices[0].message.content


links = get_links_from_init_page(scrape_result)
print(links)

# for i, link in enumerate(links):
#     scrape_result = scrape_result + fire_app.scrape_url(link, params={"formats": ["markdown"], "excludeTags": ["img", "video"]})["markdown"]

# with open(f"./dataset/article/{prompt_type}/0.txt", "w") as f:
#     f.write(scrape_result)

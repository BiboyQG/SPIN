from typing import Annotated, List, Literal
import json
import os

from pydantic import BaseModel, BeforeValidator, HttpUrl, TypeAdapter
from firecrawl import FirecrawlApp
from prompt.prof import Prof
from openai import OpenAI
from psycopg2.extras import Json
import psycopg2

fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

http_url_adapter = TypeAdapter(HttpUrl)

Url = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]


class LinkInfo(BaseModel):
    url: Url
    display_text: str


class RelatedLinks(BaseModel):
    related_links: List[LinkInfo]


class ResponseOfRelevance(BaseModel):
    answer: Literal["Yes", "No"]
    reason: str


def get_links_from_init_page(scrape_result):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format. Now you are given the initial page of the {prompt_type} article, please extract related hyperlinks that may include information about the {prompt_type} from the page according to the context of the page and the naming of the hyperlinks. For each link, provide both the URL and its display text. You should only return the JSON structure as follows: {RelatedLinks.model_json_schema()}, without any other text or comments.",
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
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format. You should only return the JSON structure as follows: {Prof.model_json_schema()}, without any other text or comments.",
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


def get_response_from_open_source_with_extra_body_update(
    scrape_result, original_response, none_keys
):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format. Now you are given an inital {prompt_type} JSON structure, please update the JSON structure with the new information from the {prompt_type} article if necessary, targeting the fields that are None or empty list or empty dict or empty string specified by the user.",
            },
            {
                "role": "user",
                "content": f"The new {prompt_type} article is: "
                + scrape_result
                + "\n"
                + f"The initial {prompt_type} JSON structure is: "
                + original_response
                + f"The fields that are None or empty list or empty dict or empty string are: {none_keys}",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": Prof.model_json_schema()},
    )
    return response.choices[0].message.content


# Append the content of the links to the original page, and then extract the information based on the appended page
def get_final_information_from_all_links_one_by_one(scrape_result, relevance_dict):
    original_response = get_response_from_open_source_with_extra_body(scrape_result)
    for i, (link, none_keys) in enumerate(relevance_dict.items()):
        original_response = get_response_from_open_source_with_extra_body_update(
            fire_app.scrape_url(
                link, params={"formats": ["markdown"], "excludeTags": ["img", "video"]}
            )["markdown"],
            original_response,
            none_keys,
        )
        print(f"Finished updating JSON structure with link {i+1}")
    with open("./results/test.json", "w") as f:
        json.dump(json.loads(original_response), f)
        print(
            "Finished extracting information from all links one by one, saved to ./results/test.json"
        )


# Connect to the Postgres database and save the information to the database
def save_prof_to_database(prof_data):
    # Database connection parameters
    db_params = {
        "dbname": os.getenv("DB_NAME", "spin"),
        "user": os.getenv("DB_USER", "admin"),
        "password": os.getenv("DB_PASSWORD", "adminpassword"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Insert query with all fields from the prof table
        insert_query = """
        INSERT INTO prof (
            fullname, title, contact, office, education, biography,
            professional_highlights, research_statement, research_interests,
            research_areas, publications, teaching_honors, research_honors,
            courses_taught
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        # Prepare the values tuple
        values = (
            prof_data["fullname"],
            prof_data["title"],
            Json(prof_data["contact"]),
            prof_data["office"],
            Json(prof_data["education"]),
            prof_data["biography"],
            Json(prof_data["professionalHighlights"]),
            prof_data["researchStatement"],
            Json(prof_data["researchInterests"]),
            Json(prof_data["researchAreas"]),
            Json(prof_data["publications"]),
            Json(prof_data["teachingHonors"]),
            Json(prof_data["researchHonors"]),
            Json(prof_data["coursesTaught"]),
        )

        # Execute the query
        cur.execute(insert_query, values)
        conn.commit()
        print(f"Successfully saved professor {prof_data['fullname']} to database")

    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def get_none_value_keys(json_obj: dict) -> List[str]:
    """
    Returns a list of keys from a JSON object where the value is None or an empty list.

    Args:
        json_obj (dict): The JSON object to check

    Returns:
        List[str]: List of keys that have None values or empty lists
    """
    return [
        key
        for key, value in json_obj.items()
        if value is None or value == [] or value == {} or value == ""
    ]


def check_link_relevance(url: str, display_text: str, none_key: str, json_data: dict) -> ResponseOfRelevance:
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at analyzing whether a hyperlink might contain information about a specific aspect of a {prompt_type}. You will analyze both the URL and its display text to make this determination. You should only answer with Yes or No and provide a brief reason, returning the JSON structure as follows: {ResponseOfRelevance.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"Given a hyperlink with:\nURL: {url}\nDisplay text: {display_text}\n\nDo you think this link might contain information about the {prompt_type} {json_data['fullname']}'s {none_key}?",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": ResponseOfRelevance.model_json_schema()},
    )
    return ResponseOfRelevance.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    with open("./dataset/article/prof/6.txt", "r") as file:
        scrape_result = file.read()

    open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    prompt_type = "prof"

    prof_data_json = get_response_from_open_source_with_extra_body(scrape_result)

    prof_data = json.loads(prof_data_json)

    none_keys = get_none_value_keys(prof_data)

    links_obj = RelatedLinks.model_validate_json(
        get_links_from_init_page(scrape_result)
    )

    relevance_dict = {}

    for link in links_obj.related_links:
        relevance_dict[link.url] = []
        for none_key in none_keys:
            relevance = check_link_relevance(link.url, link.display_text, none_key, prof_data)
            print('--------------------------------')
            if relevance.answer == "Yes":
                relevance_dict[link.url].append(none_key)
                print(f"Relevance: {relevance.answer}")
                print(f"Link {link.url} is relevant to {none_key}")
                print(f"Reason: {relevance.reason}")
            else:
                print(f"Relevance: {relevance.answer}")
                print(f"Link {link.url} is not relevant to {none_key}")
                print(f"Reason: {relevance.reason}")

    relevance_dict = {k: v for k, v in relevance_dict.items() if len(v) > 0}

    print(f"Relevance links : {relevance_dict.keys()}")

    get_final_information_from_all_links_one_by_one(scrape_result, relevance_dict)

    # # with open("test_one_by_one.json", "r") as f:
    # #     prof_data = json.load(f)
    # #     save_prof_to_database(prof_data)

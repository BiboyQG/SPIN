from typing import Annotated, List, Literal
import json
import os
from tqdm import tqdm

from pydantic import BaseModel, BeforeValidator, HttpUrl, TypeAdapter
from prompt.prof import Prof
from openai import OpenAI
from psycopg2.extras import Json
import psycopg2
from scraper import WebScraper

client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")

scraper = WebScraper()

http_url_adapter = TypeAdapter(HttpUrl)

Url = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]


class LinkInfo(BaseModel):
    url: Url
    display_text: str

    def __hash__(self):
        return hash((self.url, self.display_text))

    def __eq__(self, other):
        if not isinstance(other, LinkInfo):
            return False
        return self.url == other.url and self.display_text == other.display_text


class RelatedLinks(BaseModel):
    related_links: List[LinkInfo]


class ResponseOfRelevance(BaseModel):
    answer: Literal["Yes", "No"]
    reason: str


def get_links_from_page(scrape_result, json_data):
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} entity information in JSON format according to the content of the webpage. Now you are given the content of the {prompt_type} webpage, please extract related hyperlinks that may include information about the {prompt_type} entity from the page according to the entity's existing JSON structure and the naming(including URL and display text) of the hyperlinks. For each link that you think is relevant, provide both the URL and its display text. You should only return the JSON structure as follows: {RelatedLinks.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The content of the {prompt_type} webpage is:\n"
                + scrape_result
                + "\nThe entity's existing JSON structure is:\n"
                + json_data,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": RelatedLinks.model_json_schema()},
    )
    return response.choices[0].message.content


def get_response_from_open_source_with_extra_body(scrape_result):
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} entity information in JSON format according to the content of the webpage. You should only return the JSON structure as follows: {Prof.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The content of the {prompt_type} webpage is:\n{scrape_result}",
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
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} entity information in JSON format according to the content of the webpage. Now you are given an inital {prompt_type} JSON structure, please update the JSON structure with the new information from the {prompt_type} webpage if necessary, targeting the fields that are None or empty list or empty dict or empty string specified by the user. You should only return the JSON structure as follows: {Prof.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The new {prompt_type} webpage is:\n{scrape_result}\n"
                + f"The entity's existing JSON structure is:\n{original_response}\n"
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
    print("Getting initial response from webpage...")
    original_response = get_response_from_open_source_with_extra_body(scrape_result)

    print("\nUpdating information with relevant links...")
    for i, (link, none_keys) in enumerate(
        tqdm(relevance_dict.items(), desc="Processing links")
    ):
        print(f"\nProcessing link {i+1}/{len(relevance_dict)}: {link}")
        scrape_result = scraper.scrape_url(link)["markdown"]
        if scrape_result is None:
            print(f"Skipping link {link} because it is not accessible")
            continue
        original_response = get_response_from_open_source_with_extra_body_update(
            scrape_result,
            original_response,
            none_keys,
        )

    print("\nSaving final results...")
    with open("./results/test.json", "w") as f:
        json.dump(json.loads(original_response), f)
        print("Results saved to ./results/test.json")


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


def check_link_relevance(
    url: str, display_text: str, none_key: str, json_data: dict
) -> ResponseOfRelevance:
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at analyzing whether a hyperlink might contain information about a specific aspect of a {prompt_type}. You will analyze both the URL and its display text to make this determination. You should only answer with Yes or No and provide a brief reason, returning the JSON structure as follows: {ResponseOfRelevance.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"Given a hyperlink with:\nURL: {url}\nDisplay text: {display_text}\n\nDo you think this link might contain information about the {prompt_type} {json_data['fullname'] if json_data else 'entity'}'s {none_key}?",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": ResponseOfRelevance.model_json_schema()},
    )
    return ResponseOfRelevance.model_validate_json(response.choices[0].message.content)


def gather_links_recursively(
    initial_scrape_result, json_data, max_depth=3, visited_urls=None
):
    """
    Recursively gather links from pages up to a maximum depth.

    Args:
        initial_scrape_result (str): The initial page content
        json_data (dict): The entity's data
        max_depth (int): Maximum recursion depth to prevent infinite loops
        visited_urls (set): Set of already visited URLs to prevent cycles

    Returns:
        set: Set of LinkInfo objects containing all discovered relevant links
    """
    if visited_urls is None:
        visited_urls = set()

    if max_depth <= 0:
        return set()

    print(f"\nGathering links at depth {max_depth}...")
    all_links = set()

    # Get links from current page
    print("Extracting links from current page...")
    links_obj = RelatedLinks.model_validate_json(
        get_links_from_page(initial_scrape_result, json.dumps(json_data))
    )

    # Process each link
    for link in tqdm(links_obj.related_links, desc="Processing discovered links"):
        if link.url in visited_urls:
            continue

        visited_urls.add(link.url)
        all_links.add(link)

        try:
            print(f"\nScraping URL: {link.url}")
            new_scrape_result = scraper.scrape_url(link.url)["markdown"]

            nested_links = gather_links_recursively(
                new_scrape_result, json_data, max_depth - 1, visited_urls
            )
            all_links.update(nested_links)
        except Exception as e:
            print(f"Error processing URL {link.url}: {str(e)}")
            continue

    return all_links


if __name__ == "__main__":
    print("Starting professor information extraction process...")

    print("Reading input file...")
    with open("./dataset/article/prof/6.txt", "r") as file:
        scrape_result = file.read()

    open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    prompt_type = "prof"

    print("\nExtracting initial professor data...")
    prof_data_json = get_response_from_open_source_with_extra_body(scrape_result)
    prof_data = json.loads(prof_data_json)

    print("Identifying empty fields...")
    none_keys = get_none_value_keys(prof_data)
    print(f"Found {len(none_keys)} empty fields: {none_keys}")

    print("\nGathering initial links from page...")
    links_obj = RelatedLinks.model_validate_json(
        get_links_from_page(scrape_result, json.dumps(prof_data))
    )

    print("\nStarting recursive link discovery...")
    all_discovered_links = gather_links_recursively(
        scrape_result, prof_data, max_depth=2
    )
    links_obj.related_links = list(all_discovered_links)

    print(f"\nDiscovered {len(links_obj.related_links)} total links:")
    for link in links_obj.related_links:
        print(f"- {link.url} ({link.display_text})")

    print("\nAnalyzing link relevance...")
    relevance_dict = {}

    for link in tqdm(links_obj.related_links, desc="Checking link relevance"):
        relevance_dict[link.url] = []
        for none_key in none_keys:
            relevance = check_link_relevance(
                link.url, link.display_text, none_key, prof_data
            )
            if relevance.answer == "Yes":
                relevance_dict[link.url].append(none_key)
                print(f"\nLink {link.url} is relevant to {none_key}")
                print(f"Reason: {relevance.reason}")

    relevance_dict = {k: v for k, v in relevance_dict.items() if len(v) > 0}

    print(
        f"\nFound {len(relevance_dict)} relevant links: {list(relevance_dict.keys())}"
    )

    print("\nExtracting information from relevant links...")
    get_final_information_from_all_links_one_by_one(scrape_result, relevance_dict)

    print("\nProcess completed successfully!")

from typing import Annotated, List, Literal
import json
import os
from tqdm import tqdm
import logging
import argparse

from pydantic import BaseModel, BeforeValidator, HttpUrl, TypeAdapter
from prompt.prof import Prof
from openai import OpenAI
# from psycopg2.extras import Json
# import psycopg2
from scraper import WebScraper

# client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
client = OpenAI(base_url="http://localhost:8000/v1")

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


def setup_logging(open_source_model, prompt_type, max_depth):
    # Create logs directory structure
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(f"./logs/{open_source_model}", exist_ok=True)
    os.makedirs(f"./logs/{open_source_model}/{prompt_type}", exist_ok=True)
    os.makedirs(f"./logs/{open_source_model}/{prompt_type}/{max_depth}", exist_ok=True)

    log_path = f"./logs/{open_source_model}/{prompt_type}/{max_depth}/log.txt"

    # Disable httpx logger to prevent OpenAI HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure logging with a more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s\n",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    # Add custom log formatting functions
    def log_section(message):
        logging.info(f"\n{'='*80}\n{message}\n{'='*80}")

    def log_subsection(message):
        logging.info(f"\n{'-'*40}\n{message}\n{'-'*40}")

    def log_url_processing(url, index, total):
        logging.info(f"\n{'#'*80}\nProcessing URL [{index}/{total}]: {url}\n{'#'*80}")

    # Add these functions to the logging module for easy access
    logging.section = log_section
    logging.subsection = log_subsection
    logging.url_processing = log_url_processing


def get_links_from_page(scrape_result, json_data):
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} entity information in JSON format according to the content of the webpage. Now you are given the content of the {prompt_type} webpage, please extract related hyperlinks that may include information about the {prompt_type} entity from the page according to the entity's existing JSON structure and the naming(including URL and display text) of the hyperlinks. For each link that you think may contain the information of the entity's field, provide both the URL and its display text. You should return them only in the JSON structure as follows: {RelatedLinks.model_json_schema()}, without any other text or comments.",
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
def get_final_information_from_all_links_one_by_one(
    scrape_result, relevance_dict, output_path
):
    logging.info("Getting initial response from webpage...")
    original_response = get_response_from_open_source_with_extra_body(scrape_result)

    logging.info("\nUpdating information with relevant links...")
    for i, (link, none_keys) in enumerate(
        tqdm(relevance_dict.items(), desc="Processing links")
    ):
        logging.info(f"\nProcessing link {i+1}/{len(relevance_dict)}: {link}")
        scrape_result = scraper.scrape_url(link)["markdown"]
        if scrape_result is None:
            logging.warning(f"Skipping link {link} because it is not accessible")
            continue
        original_response = get_response_from_open_source_with_extra_body_update(
            scrape_result,
            original_response,
            none_keys,
        )

    logging.info("\nSaving final results...")
    with open(output_path, "w") as f:
        json.dump(json.loads(original_response), f)
        logging.info(f"Results saved to {output_path}")


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
        logging.info(
            f"Successfully saved professor {prof_data['fullname']} to database"
        )

    except Exception as e:
        logging.error(f"Error saving to database: {str(e)}")
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
    initial_scrape_result,
    json_data,
    none_keys,
    max_depth=3,
    visited_urls=None,
    relevance_dict=None,
):
    if visited_urls is None:
        visited_urls = set()
    if relevance_dict is None:
        relevance_dict = {}

    if max_depth <= 0:
        return set(), relevance_dict

    logging.section(f"Gathering links at depth {max_depth}")
    relevant_links = set()

    # Get links from current page
    logging.subsection("Extracting links from current page")
    links_obj = RelatedLinks.model_validate_json(
        get_links_from_page(initial_scrape_result, json.dumps(json_data))
    )

    # Process each link
    total_links = len(links_obj.related_links)
    logging.info(f"Found {total_links} links to process")

    for idx, link in enumerate(
        tqdm(links_obj.related_links, desc="Processing discovered links")
    ):
        logging.subsection(f"Processing link {idx + 1}/{total_links}")
        logging.info(f"URL: {link.url}")
        logging.info(f"Display text: {link.display_text}")

        if link.url in visited_urls:
            logging.info("⏩ Skipping - URL already visited")
            continue

        if link.url.lower().endswith(".pdf"):
            logging.info("⏩ Skipping - PDF URL")
            continue

        visited_urls.add(link.url)

        # Check relevance of the link
        relevant_fields = []
        logging.info("Checking relevance for empty fields:")

        for none_key in none_keys:
            relevance = check_link_relevance(
                link.url, link.display_text, none_key, json_data
            )
            if relevance.answer == "Yes":
                relevant_fields.append(none_key)
                logging.info(f"✅ Relevant to '{none_key}': {relevance.reason}")
            else:
                logging.info(f"❌ Not relevant to '{none_key}': {relevance.reason}")

        # Only process and recurse on relevant links
        if relevant_fields:
            relevant_links.add(link)
            relevance_dict[link.url] = relevant_fields
            logging.info(
                f"📍 Link is relevant for fields: {', '.join(relevant_fields)}"
            )

            try:
                logging.subsection(f"Scraping relevant URL: {link.url}")
                new_scrape_result = scraper.scrape_url(link.url)["markdown"]

                if new_scrape_result is None:
                    logging.warning("⚠️  URL not accessible - skipping")
                    continue

                logging.info("🔄 Starting recursive link gathering...")
                nested_links, nested_relevance = gather_links_recursively(
                    new_scrape_result,
                    json_data,
                    none_keys,
                    max_depth - 1,
                    visited_urls,
                    relevance_dict,
                )
                relevant_links.update(nested_links)
                logging.info(f"Found {len(nested_links)} additional relevant links")

            except Exception as e:
                logging.error(f"❌ Error processing URL: {str(e)}")
                continue
        else:
            logging.info("⏩ Skipping - No relevant fields found")

    logging.section(f"Completed depth {max_depth}")
    logging.info(f"Total relevant links found at this depth: {len(relevant_links)}")

    return relevant_links, relevance_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process entity information with configurable depth."
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=2,
        help="Maximum depth for recursive link gathering (default: 2)",
    )
    args = parser.parse_args()

    open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    prompt_type = "prof"
    max_depth = args.depth

    # Set up logging
    setup_logging(open_source_model, prompt_type, max_depth)

    logging.section("Starting professor information extraction process")

    logging.info("Reading URLs from prof.txt...")
    with open("./dataset/source/prof.txt", "r") as file:
        urls = [url.strip() for url in file.readlines()]

    # Process each URL
    for idx, url in enumerate(urls):
        logging.url_processing(url, idx + 1, len(urls))

        logging.subsection("Scraping webpage content")
        try:
            scrape_result = scraper.scrape_url(url)["markdown"]
            if scrape_result is None:
                logging.warning(f"⚠️  Skipping URL {url} - unable to scrape content")
                continue
        except Exception as e:
            logging.error(f"❌ Error scraping URL {url}: {str(e)}")
            continue

        logging.subsection("Extracting initial professor data")
        prof_data_json = get_response_from_open_source_with_extra_body(scrape_result)
        prof_data = json.loads(prof_data_json)

        logging.subsection("Identifying empty fields")
        none_keys = get_none_value_keys(prof_data)
        logging.info(f"Found {len(none_keys)} empty fields:")
        for key in none_keys:
            logging.info(f"  • {key}")

        logging.subsection("Gathering relevant links recursively")
        all_discovered_links, relevance_dict = gather_links_recursively(
            scrape_result, prof_data, none_keys, max_depth=2
        )

        logging.info(f"\nDiscovered {len(all_discovered_links)} relevant links:")
        for link in all_discovered_links:
            logging.info(f"  • {link.url}")
            logging.info(f"    ├─ Display: {link.display_text}")
            logging.info(f"    └─ Relevant to: {', '.join(relevance_dict[link.url])}")

        logging.subsection("Extracting information from relevant links")
        # Create results directory if it doesn't exist
        os.makedirs(
            f"./results/{open_source_model}/{prompt_type}/{max_depth}", exist_ok=True
        )
        output_path = (
            f"./results/{open_source_model}/{prompt_type}/{max_depth}/{idx}.json"
        )

        get_final_information_from_all_links_one_by_one(
            scrape_result, relevance_dict, output_path
        )

        logging.info(f"✅ Results saved to {output_path}")

    logging.section("Process completed successfully!")

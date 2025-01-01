from typing import List, Literal
from datetime import datetime
from tqdm import tqdm
import argparse
import requests
import json
import logging
import time
import csv
import sys
import os

from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError
from scraper import WebScraper
from prompt.prof import Prof
from openai import OpenAI

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))

scraper = WebScraper()

http_url_adapter = TypeAdapter(HttpUrl)


def try_validate_url(value: str) -> str:
    """Validates a URL string, returning the original string if validation fails."""
    try:
        return str(http_url_adapter.validate_python(value))
    except ValidationError as e:
        logging.warning(f"Invalid URL format: {value}")
        return value  # Return the original value instead of None


class LinkInfo(BaseModel):
    url: str  # Change from Url to str to accept any string initially
    display_text: str

    def __hash__(self):
        return hash((self.url, self.display_text))

    def __eq__(self, other):
        if not isinstance(other, LinkInfo):
            return False
        return self.url == other.url and self.display_text == other.display_text

    def is_valid_url(self) -> bool:
        """Check if the URL is valid."""
        try:
            http_url_adapter.validate_python(self.url)
            return True
        except ValidationError:
            return False


class RelatedLinks(BaseModel):
    related_links: List[LinkInfo]

    @property
    def valid_links(self) -> List[LinkInfo]:
        """Returns only the links with valid URLs."""
        return [link for link in self.related_links if link.is_valid_url()]


class ResponseOfRelevance(BaseModel):
    answer: Literal["Yes", "No"]
    reason: str


def setup_logging(open_source_model, prompt_type, max_depth):
    # Create results directory structure instead of logs
    base_path = f"./results/{open_source_model}/{prompt_type}/{max_depth}"
    os.makedirs(base_path, exist_ok=True)

    log_path = f"{base_path}/process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

    def log_summary(url, duration, relevant_links_info, update_times=None):
        summary = f"\n📊 Summary for {url}\n"
        summary += f"{'─'*80}\n"
        summary += f"⏱️  Total Processing Duration: {duration:.2f} seconds\n"

        if relevant_links_info:
            summary += "\n📎 Relevant Links Analysis:\n"
            for link_url, info in relevant_links_info.items():
                summary += f"\n🔗 {link_url}\n"
                formatted_lines = format_relevant_link_info(
                    link_url, info, update_times
                )
                summary += "\n".join(formatted_lines) + "\n"
        else:
            summary += "\n❌ No relevant links found\n"

        summary += f"{'─'*80}\n"
        logging.info(summary)

    # Add these functions to the logging module
    logging.section = log_section
    logging.subsection = log_subsection
    logging.url_processing = log_url_processing
    logging.summary = log_summary


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
                "content": f"You are an expert at summarizing {prompt_type} entity information in JSON format according to the content of the webpage. Now you are given an inital {prompt_type} JSON structure, please update the JSON structure with the new information from the {prompt_type} webpage if necessary, targeting the fields that are None or empty list or empty dict or empty string specified by the user. If there are too much information that can be updated to one field(value of this field is a list), you should only update the field with at most 25 numbers of information. You should only return the JSON structure as follows: {Prof.model_json_schema()}, without any other text or comments.",
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
    update_times = {}  # Track update times for each link

    for i, (link, none_keys) in enumerate(
        tqdm(relevance_dict.items(), desc="Processing links")
    ):
        logging.info(f"\nProcessing link {i+1}/{len(relevance_dict)}: {link}")
        scrape_result = scraper.scrape_url(link)["markdown"]
        if scrape_result is None:
            logging.warning(f"Skipping link {link} because it is not accessible")
            continue

        # Track time for JSON update
        start_time = time.time()
        original_response = get_response_from_open_source_with_extra_body_update(
            scrape_result,
            original_response,
            none_keys,
        )
        update_times[link] = time.time() - start_time

    logging.info("\nSaving final results...")
    with open(output_path, "w") as f:
        json.dump(json.loads(original_response), f)
        logging.info(f"Results saved to {output_path}")

    return update_times


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
    valid_links = links_obj.valid_links

    # Process each link
    total_links = len(valid_links)
    logging.info(f"Found {total_links} links to process")

    for idx, link in enumerate(tqdm(valid_links, desc="Processing discovered links")):
        logging.subsection(f"Processing link {idx + 1}/{total_links}")
        logging.info(f"URL: {link.url}")
        logging.info(f"Display text: {link.display_text}")

        if link.url in visited_urls:
            logging.info("⏩ Skipping - URL already visited")
            continue

        if link.url.lower().endswith(".pdf"):
            logging.info("⏩ Skipping - PDF URL")
            continue

        if "arxiv" in link.url:
            logging.info("⏩ Skipping - arxiv URL")
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


def format_relevant_link_info(link_url, info, update_times=None, indent="   "):
    """Format relevant link information including fields, reasons, and update time.

    Args:
        link_url: URL of the relevant link
        info: Dictionary containing fields and reasons
        update_times: Dictionary of update times for links (optional)
        indent: String for indentation (default: "   ")

    Returns:
        list: List of formatted strings for the link information
    """
    formatted_lines = []
    formatted_lines.append(f"{indent}Fields: {', '.join(info['fields'])}")

    if "reasons" in info:
        formatted_lines.append(f"{indent}Reasons:")
        for field, reason in info["reasons"].items():
            formatted_lines.append(f"{indent}• {field}: {reason}")

    if update_times and link_url in update_times:
        formatted_lines.append(
            f"{indent}⏱️  Update Time: {update_times.get(link_url, 0):.2f} seconds"
        )

    return formatted_lines


def write_process_stats_to_csv(
    open_source_model: str,
    prompt_type: str,
    max_depth: int,
    url_processing_times: dict,
    url_relevant_links: dict,
    all_update_times: dict,
):
    """
    Write processing statistics to a CSV file.

    Args:
        open_source_model: Name of the model used
        prompt_type: Type of prompt used
        max_depth: Maximum depth of link traversal
        url_processing_times: Dictionary of processing times for each initial URL
        url_relevant_links: Dictionary of relevant links and their details for each initial URL
        all_update_times: Dictionary of update times for relevant links
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"./results/{open_source_model}/{prompt_type}/{max_depth}/process_stats_{timestamp}.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(
            [
                "Initial URL",
                "Processing Time (s)",
                "Relevant Link",
                "Relevant Fields",
                "Relevance Reasons",
                "Update Time (s)",
            ]
        )

        # Write data for each initial URL
        for initial_url, processing_time in url_processing_times.items():
            relevant_links = url_relevant_links.get(initial_url, {})
            update_times = all_update_times.get(initial_url, {})

            if not relevant_links:
                # Write row for URLs with no relevant links
                writer.writerow([initial_url, f"{processing_time:.2f}", "", "", "", ""])
            else:
                # Write rows for each relevant link
                for rel_url, info in relevant_links.items():
                    fields = ", ".join(info["fields"])
                    reasons = "; ".join(
                        [
                            f"{field}: {reason}"
                            for field, reason in info["reasons"].items()
                        ]
                    )
                    update_time = update_times.get(rel_url, 0)

                    writer.writerow(
                        [
                            initial_url,
                            f"{processing_time:.2f}",
                            rel_url,
                            fields,
                            reasons,
                            f"{update_time:.2f}",
                        ]
                    )

    logging.info(f"📊 Process statistics written to: {csv_path}")


def brave_search(query: str) -> str:
    """
    Perform a Brave search and return the top result URL.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    base_url = os.getenv("BRAVE_SEARCH_URL")

    if not api_key or not base_url:
        raise ValueError(
            "Brave Search API key or URL not found in environment variables"
        )

    headers = {"X-Subscription-Token": api_key}
    params = {"q": query}

    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()

        if results.get("web") and results["web"].get("results"):
            return results["web"]["results"][0]["url"]
        else:
            raise ValueError("No search results found")
    except Exception as e:
        logging.error(f"Error performing Brave search: {str(e)}")
        raise


def process_input_urls(input_str: str) -> List[str]:
    """
    Process the input string to return a list of URLs.
    If input is a query, performs a Brave search and returns the top result.
    If input is a comma-separated list of URLs, splits and returns them.
    If input is a single URL, returns it as a single-element list.
    """
    # Check if input contains any URLs
    if "http://" in input_str or "https://" in input_str:
        # Split by comma if multiple URLs
        urls = [url.strip() for url in input_str.split(",")]
        return urls
    else:
        # Treat as search query
        logging.info(f"Performing Brave search for query: {input_str}")
        top_url = brave_search(input_str)
        logging.info(f"Found top result URL: {top_url}")
        return [top_url]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process entity information with configurable depth."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input query or URL(s). Can be a search query, single URL, or comma-separated URLs",
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

    try:
        urls = process_input_urls(args.input)
        logging.info(f"Processing URLs: {urls}")
    except Exception as e:
        logging.error(f"Failed to process input: {str(e)}")
        sys.exit(1)

    start_time = time.time()
    url_processing_times = {}
    url_relevant_links = {}
    all_update_times = {}  # New dictionary to store update times for all URLs

    # Process each URL
    for idx, url in enumerate(urls):
        url_start_time = time.time()
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

        os.makedirs(
            f"./results/{open_source_model}/{prompt_type}/{max_depth}", exist_ok=True
        )
        output_path = (
            f"./results/{open_source_model}/{prompt_type}/{max_depth}/{prof_data['fullname'].replace(' ', '_').lower()}.json"
        )

        if len(none_keys) != 0:
            logging.subsection("Gathering relevant links recursively")
            all_discovered_links, relevance_dict = gather_links_recursively(
                scrape_result, prof_data, none_keys, max_depth=max_depth
            )

            # Store relevant links info for summary
            url_relevant_links[url] = {
                link_url: {
                    "fields": fields,
                    "reasons": {
                        field: check_link_relevance(
                            link_url,
                            next(
                                link.display_text
                                for link in all_discovered_links
                                if link.url == link_url
                            ),
                            field,
                            prof_data,
                        ).reason
                        for field in fields
                    },
                }
                for link_url, fields in relevance_dict.items()
            }

            logging.info(f"\nDiscovered {len(all_discovered_links)} relevant links:")
            for link in all_discovered_links:
                logging.info(f"  • {link.url}")
                logging.info(f"    ├─ Display: {link.display_text}")
                logging.info(
                    f"    └─ Relevant to: {', '.join(relevance_dict[link.url])}"
                )

            logging.subsection("Extracting information from relevant links")

            update_times = get_final_information_from_all_links_one_by_one(
                scrape_result, relevance_dict, output_path
            )
            all_update_times[url] = update_times  # Store update times for this URL
        else:
            logging.subsection("No empty fields found, saving initial data")
            with open(output_path, "w") as f:
                json.dump(prof_data, f)
            all_update_times[url] = {}  # Empty dict for URLs with no updates

        url_processing_times[url] = time.time() - url_start_time
        logging.summary(
            url,
            url_processing_times[url],
            url_relevant_links.get(url, {}),
            all_update_times.get(url, {}),
        )
        logging.info(f"✅ Results saved to {output_path}")

    total_duration = time.time() - start_time

    logging.section("Final Process Summary")
    logging.info(f"Total Processing Time: {total_duration:.2f} seconds")
    logging.info(f"Number of URLs Processed: {len(urls)}")

    logging.info("\n📊 Detailed Analysis by URL:")
    for url in urls:
        logging.info(f"\n{'─'*80}")
        logging.info(f"🌐 {url}")
        logging.info(
            f"⏱️  Processing Time: {url_processing_times.get(url, 0):.2f} seconds"
        )

        relevant_links = url_relevant_links.get(url, {})
        if relevant_links:
            logging.info("\n📎 Relevant Links Found:")
            for link_url, info in relevant_links.items():
                logging.info(f"\n  🔗 {link_url}")
                formatted_lines = format_relevant_link_info(
                    link_url, info, all_update_times.get(url, {}), indent="     "
                )
                for line in formatted_lines:
                    logging.info(line)
        else:
            logging.info("\n❌ No relevant links found for this URL")

        logging.info(f"{'─'*80}")

    logging.section("Writing process statistics to CSV")
    write_process_stats_to_csv(
        open_source_model,
        prompt_type,
        max_depth,
        url_processing_times,
        url_relevant_links,
        all_update_times,
    )

    logging.section("✅ Process completed successfully!")

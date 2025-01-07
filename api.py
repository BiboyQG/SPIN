from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, List
from pydantic import BaseModel
import json
import uuid
import os
import argparse
import logging
from datetime import datetime
import requests

# Import existing functionality
from multi import (
    ResponseOfRelevance,
    RelatedLinks,
)
from schema.schema_manager import schema_manager
from scraper import WebScraper
from openai import OpenAI
from tqdm import tqdm
import uvicorn


global client
client = None


class MemoryLogger:
    def __init__(self):
        self.logs: List[Dict] = []
        self.level = logging.INFO

    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append({"timestamp": timestamp, "level": level, "message": message})

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)

    def section(self, message: str):
        self._log("INFO", f"\n{'='*60}\n{message}\n{'='*60}")

    def subsection(self, message: str):
        self._log("INFO", f"\n{'-'*60}\n{message}\n{'-'*60}")

    def url_processing(self, url: str, index: int, total: int):
        self._log(
            "INFO", f"\n{'#'*60}\nProcessing URL [{index}/{total}]: {url}\n{'#'*60}"
        )

    def summary(
        self,
        url: str,
        duration: float,
        relevant_links_info: Dict,
        update_times: Optional[Dict] = None,
    ):
        summary_text = f"\n📊 Summary for {url}\n"
        summary_text += f"{'─'*60}\n"
        summary_text += f"⏱️  Total Processing Duration: {duration:.2f} seconds\n"

        if relevant_links_info:
            summary_text += "\n📎 Relevant Links Analysis:\n"
            for link_url, info in relevant_links_info.items():
                summary_text += f"\n🔗 {link_url}\n"
                if "fields" in info:
                    summary_text += f"   Fields: {', '.join(info['fields'])}\n"
                if "reasons" in info:
                    summary_text += "   Reasons:\n"
                    for field, reason in info["reasons"].items():
                        summary_text += f"   • {field}: {reason}\n"
                if update_times and link_url in update_times:
                    summary_text += (
                        f"   ⏱️  Update Time: {update_times[link_url]:.2f} seconds\n"
                    )
        else:
            summary_text += "\n❌ No relevant links found\n"

        summary_text += f"{'─'*60}\n"
        self._log("INFO", summary_text)

    def get_logs(self) -> List[Dict]:
        return self.logs


class ExtractionRequest(BaseModel):
    input: str  # Search query or URLs
    depth: int = 1
    openai_base_url: str
    model_name: str
    schema_type: Optional[str] = None  # Optional schema type for manual input


class ExtractionResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[Dict] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    logs: Optional[List[Dict]] = None


class ExtractionTask:
    def __init__(self, request: ExtractionRequest):
        self.request = request
        self.status = "pending"
        self.progress = {
            "stage": "initializing",  # Current stage of extraction
            "stage_progress": 0,  # Progress within current stage (0-100)
            "current_url": None,  # Current URL being processed
            "url_number": 0,  # Current URL number
            "total_urls": 0,  # Total URLs to process
            "message": "Starting extraction...",  # Current status message
            "schema_type": None,  # Schema type if manually provided
        }
        self.result = None
        self.error = None
        self.logger = MemoryLogger()

    def update_progress(self, stage: str, message: str, stage_progress: int = 0):
        self.progress.update(
            {"stage": stage, "stage_progress": stage_progress, "message": message}
        )

    def run(self):
        global client

        try:
            self.status = "running"

            # Configure environment
            os.environ["OPENAI_BASE_URL"] = self.request.openai_base_url
            os.environ["OPEN_SOURCE_MODEL"] = self.request.model_name

            client = OpenAI()

            # Process URLs
            urls = process_input_urls(self.request.input)
            self.progress["total_urls"] = len(urls)

            results = {}
            for idx, url in enumerate(urls):
                self.progress.update(
                    {
                        "current_url": url,
                        "url_number": idx + 1,
                    }
                )

                # Stage 1: Scrape content
                self.update_progress("scraping", f"Scraping content from {url}", 0)
                scrape_result = scraper.scrape_url(url)["markdown"]
                if not scrape_result:
                    self.logger.warning(f"Failed to scrape URL: {url}")
                    continue

                # Stage 2: Detect schema
                self.update_progress(
                    "schema_detection",
                    "Analyzing webpage content to detect schema...",
                    20,
                )

                # Use provided schema type or detect it
                if self.request.schema_type:
                    schema_type = self.request.schema_type
                    self.update_progress(
                        "schema_detection",
                        f"Using provided schema: {snake_case_to_normal_case(schema_type)}",
                        40,
                    )
                else:
                    schema_result = detect_schema(scrape_result)
                    if schema_result.schema == "No match":
                        self.update_progress(
                            "schema_detection",
                            "The schema of the entity doesn't match any existing schema, please input the general schema name of the entity below:",
                            30,
                        )
                        return
                    schema_type = schema_result.schema
                    self.update_progress(
                        "schema_detection",
                        f"Detected schema: {snake_case_to_normal_case(schema_type)}. Reason: {schema_result.reason}",
                        40,
                    )

                # Stage 3: Generate schema if needed
                is_generated = False
                if (
                    self.request.schema_type
                    and schema_type not in schema_manager.get_schema_names()
                ):
                    is_generated = True
                    self.update_progress(
                        "schema_generation",
                        f"Generating new schema for type: {snake_case_to_normal_case(schema_type)}",
                        40,
                    )
                    new_schema_code = generate_new_schema(scrape_result, schema_type)
                    schema_manager.save_new_schema(schema_type, new_schema_code)

                # Stage 4: Extract initial data
                self.update_progress(
                    "initial_extraction",
                    f"Extracting initial entity data for schema: {snake_case_to_normal_case(schema_type)}. The reason is: {schema_result.reason}" if not is_generated else f"Extracting initial entity data for newly generated schema: {snake_case_to_normal_case(schema_type)}",
                    50,
                )
                original_response = get_response_from_open_source_with_extra_body(
                    scrape_result, schema_type
                )
                entity_data = json.loads(original_response)

                # Stage 5: Get empty fields
                self.update_progress("analyzing_fields", "Analyzing empty fields", 60)
                none_keys = get_none_value_keys(entity_data)

                if none_keys:
                    # Stage 6: Gather links
                    self.update_progress(
                        "gathering_links",
                        f"Gathering relevant links for fields: {', '.join(none_keys)}",
                        70,
                    )
                    all_discovered_links, relevance_dict = gather_links_recursively(
                        scrape_result,
                        entity_data,
                        none_keys,
                        schema_type,
                        max_depth=self.request.depth,
                    )

                    # Stage 7: Update information
                    total_links = len(relevance_dict)
                    if total_links > 0:
                        progress_per_link = (
                            20 / total_links
                        )  # 20 is the range from 80 to 100
                        for i, (link, fields) in enumerate(relevance_dict.items()):
                            current_progress = 80 + (i * progress_per_link)
                            self.update_progress(
                                "updating_data",
                                f"Updating information from link {i+1}/{total_links}: {link}",
                                int(current_progress),
                            )
                            scrape_result = scraper.scrape_url(link)["markdown"]
                            if scrape_result is None:
                                continue

                            original_response = (
                                get_response_from_open_source_with_extra_body_update(
                                    scrape_result,
                                    original_response,
                                    fields,
                                    schema_type,
                                )
                            )

                    results[url] = original_response
                else:
                    results[url] = original_response

                self.update_progress(
                    "finalizing", f"Completed processing URL {idx + 1}/{len(urls)}", 100
                )

            self.result = results
            self.status = "completed"
            self.update_progress("completed", "Extraction completed successfully", 100)

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.logger.error(str(e))
            self.update_progress("failed", f"Extraction failed: {str(e)}", 0)
            raise


app = FastAPI()
scraper = WebScraper()
logger = MemoryLogger()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active tasks
tasks: Dict[str, ExtractionTask] = {}


@app.post("/extract", response_model=ExtractionResponse)
def start_extraction(request: ExtractionRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task = ExtractionTask(request)
    tasks[task_id] = task

    background_tasks.add_task(task.run)

    return ExtractionResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=ExtractionResponse)
def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return ExtractionResponse(
        task_id=task_id,
        status=task.status,
        progress=task.progress,
        result=task.result,
        error=task.error,
        logs=task.logger.get_logs(),  # Include logs in the response
    )


# Additional endpoints for settings and schema management
@app.get("/schemas")
def get_schemas():
    return {"schemas": schema_manager.get_schema_names()}


@app.post("/schemas/{name}")
def create_schema(name: str, content: str):
    try:
        schema_manager.save_new_schema(name, content)
        return {"message": f"Schema {name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def snake_case_to_camel_case(text: str) -> str:
    """Convert snake case to camel case."""
    return "".join(word.capitalize() for word in text.split("_"))


def snake_case_to_normal_case(text: str) -> str:
    """Convert snake case to normal case."""
    return " ".join(word.capitalize() for word in text.split("_"))


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
        logger.error(f"Error performing Brave search: {str(e)}")
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
        top_url = brave_search(input_str)
        return [top_url]


def detect_schema(webpage_content: str):
    """Detect the appropriate schema for webpage content using LLM."""
    available_schemas = schema_manager.get_schema_names()
    ResponseOfSchema = schema_manager.get_response_of_schema()

    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at analyzing webpage content and determining the type of entity being described. You will analyze the content and determine if it matches one of the following schemas: {', '.join(available_schemas)}. Return your analysis as a JSON object with the matched schema name and reason. If no schema matches, return 'No match'. You should only return the JSON structure as follows: {ResponseOfSchema.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"Analyze this webpage content and determine which schema it matches:\n{webpage_content}",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": ResponseOfSchema.model_json_schema()},
    )

    return ResponseOfSchema.model_validate_json(response.choices[0].message.content)


def generate_new_schema(webpage_content: str, schema_type: str) -> str:
    """Generate a new Pydantic schema based on webpage content using LLM."""
    with open("./schema/professor.py", "r") as f:
        example_schema = f.read()
    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at creating Pydantic schemas for different types of entities. Given webpage content and the desired schema type by the user, generate a complete Pydantic schema that captures all relevant information about the entity being described. The schema should:
- Use appropriate field types and nested models
- Include field descriptions using Field(description="...") when necessary
- Be general enough while incorporating key aspects from the example.
- Follow similar structure to this example:

{example_schema}

Return only the Python code for the schema, without any other text(```python and ``` are forbidden, just return pure Python code).""",
            },
            {
                "role": "user",
                "content": f"Generate a {snake_case_to_normal_case(schema_type)} Pydantic schema, whose model name should be exactly {snake_case_to_camel_case(schema_type)}, for this webpage content:\n{webpage_content}",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
    )

    return response.choices[0].message.content


def get_response_from_open_source_with_extra_body(scrape_result: str, schema_type: str):
    """Get entity information using the appropriate schema."""
    logger.info(f"Getting response from open source with extra body for {schema_type}")
    entity_schema = schema_manager.get_schema(schema_type)
    logger.info(f"Entity schema: {entity_schema}")

    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {schema_type} entity information in JSON format according to the content of the webpage. You should only return the JSON structure as follows: {entity_schema.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The content of the webpage is:\n{scrape_result}",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": entity_schema.model_json_schema()},
    )
    return response.choices[0].message.content


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
    url: str,
    display_text: str,
    none_key: str,
    json_data: dict,
    schema_type: str,
) -> ResponseOfRelevance:
    entity_name = (
        json_data.get("name")
        or json_data.get("fullname")
        or json_data.get("code")
        or json_data.get("title")
    )
    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at analyzing whether a hyperlink might contain information about a specific aspect of a {schema_type}. You will analyze both the URL and its display text to make this determination. You should only answer with Yes or No and provide a brief reason, returning the JSON structure as follows: {ResponseOfRelevance.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"Given a hyperlink with:\nURL: {url}\nDisplay text: {display_text}\n\nDo you think this link might contain information about the {schema_type} {entity_name}'s {none_key}?",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": ResponseOfRelevance.model_json_schema()},
    )
    return ResponseOfRelevance.model_validate_json(response.choices[0].message.content)


def get_links_from_page(scrape_result, json_data, schema_type: str):
    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {schema_type} entity information in JSON format according to the content of the webpage. Now you are given the content of the {schema_type} webpage, please extract related hyperlinks that may include information about the {schema_type} entity from the page according to the entity's existing JSON structure and the naming(including URL and display text) of the hyperlinks. For each link that you think may contain the information of the entity's field, provide both the URL and its display text. You should return them only in the JSON structure as follows: {RelatedLinks.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The content of the {schema_type} webpage is:\n"
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


def gather_links_recursively(
    initial_scrape_result,
    json_data,
    none_keys,
    schema_type: str,
    max_depth=3,
    visited_urls=None,
    relevance_dict=None,
):
    global logger
    if visited_urls is None:
        visited_urls = set()
    if relevance_dict is None:
        relevance_dict = {}

    if max_depth <= 0:
        return set(), relevance_dict

    logger.section(f"Gathering links at depth {max_depth}")
    relevant_links = set()

    # Get links from current page
    logger.subsection("Extracting links from current page")
    links_obj = RelatedLinks.model_validate_json(
        get_links_from_page(initial_scrape_result, json.dumps(json_data), schema_type)
    )
    valid_links = links_obj.valid_links

    # Process each link
    total_links = len(valid_links)
    logger.info(f"Found {total_links} links to process")

    for idx, link in enumerate(tqdm(valid_links, desc="Processing discovered links")):
        logger.subsection(f"Processing link {idx + 1}/{total_links}")
        logger.info(f"URL: {link.url}")
        logger.info(f"Display text: {link.display_text}")

        if link.url in visited_urls:
            logger.info("⏩ Skipping - URL already visited")
            continue

        if link.url.lower().endswith(".pdf"):
            logger.info("⏩ Skipping - PDF URL")
            continue

        if "arxiv" in link.url:
            logger.info("⏩ Skipping - arxiv URL")
            continue

        if link.url.lower().endswith(".txt"):
            logger.info("⏩ Skipping - txt URL")
            continue

        if "drive.google" in link.url:
            logger.info("⏩ Skipping - google drive URL")
            continue

        visited_urls.add(link.url)

        # Check relevance of the link
        relevant_fields = []
        logger.info("Checking relevance for empty fields:")

        for none_key in none_keys:
            relevance = check_link_relevance(
                link.url, link.display_text, none_key, json_data, schema_type
            )
            if relevance.answer == "Yes":
                relevant_fields.append(none_key)
                logger.info(f"✅ Relevant to '{none_key}': {relevance.reason}")
            else:
                logger.info(f"❌ Not relevant to '{none_key}': {relevance.reason}")

        # Only process and recurse on relevant links
        if relevant_fields:
            relevant_links.add(link)
            relevance_dict[link.url] = relevant_fields
            logger.info(f"📍 Link is relevant for fields: {', '.join(relevant_fields)}")

            try:
                logger.subsection(f"Scraping relevant URL: {link.url}")
                new_scrape_result = scraper.scrape_url(link.url)["markdown"]

                if new_scrape_result is None:
                    logger.warning("⚠️  URL not accessible - skipping")
                    continue

                logger.info("🔄 Starting recursive link gathering...")
                nested_links, nested_relevance = gather_links_recursively(
                    new_scrape_result,
                    json_data,
                    none_keys,
                    schema_type,
                    max_depth - 1,
                    visited_urls,
                    relevance_dict,
                )
                relevant_links.update(nested_links)
                logger.info(f"Found {len(nested_links)} additional relevant links")

            except Exception as e:
                logger.error(f"❌ Error processing URL: {str(e)}")
                continue
        else:
            logger.info("⏩ Skipping - No relevant fields found")

    logger.section(f"Completed depth {max_depth}")
    logger.info(f"Total relevant links found at this depth: {len(relevant_links)}")

    return relevant_links, relevance_dict


def get_response_from_open_source_with_extra_body_update(
    scrape_result,
    original_response,
    none_keys,
    schema_type: str,
):
    entity_schema = schema_manager.get_schema(schema_type)

    response = client.chat.completions.create(
        model=os.getenv("OPEN_SOURCE_MODEL"),
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {schema_type} entity information in JSON format according to the content of the webpage. Now you are given an inital {schema_type} JSON structure, please update the JSON structure with the new information from the {schema_type} webpage if necessary, targeting the fields that are None or empty list or empty dict or empty string specified by the user. If there are too much information that can be updated to one field(value of this field is a list), you should only update the field with at most 25 numbers of information. You should only return the JSON structure as follows: {entity_schema.model_json_schema()}, without any other text or comments.",
            },
            {
                "role": "user",
                "content": f"The new {schema_type} webpage is:\n{scrape_result}\n"
                + f"The entity's existing JSON structure is:\n{original_response}\n"
                + f"The fields that are None or empty list or empty dict or empty string are: {none_keys}",
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        extra_body={"guided_json": entity_schema.model_json_schema()},
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Information Extraction API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to run the server on",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run server in debug mode")

    args = parser.parse_args()

    # Configure logging
    logging_config = uvicorn.config.LOGGING_CONFIG
    if args.debug:
        logging_config["loggers"]["uvicorn"]["level"] = "DEBUG"
    else:
        logging_config["loggers"]["uvicorn"]["level"] = "INFO"

    # Run the server
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_config=logging_config,
    )

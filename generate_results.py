from functools import partial
import concurrent.futures
from openai import OpenAI
import pathlib as pl
import instructor
import importlib
import json
import os

open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"
openai_model = "gpt-4o-mini"
prompt_type = "movie"

prompt_module = importlib.import_module(f"prompt.{prompt_type}")

dataset_path = pl.Path(f"dataset/article/{prompt_type}")
open_model_results_path = pl.Path(f"dataset/results/open-source/{open_source_model}/{prompt_type}")
open_model_instructor_results_path = open_model_results_path / "instructor"
proprietary_results_path = pl.Path(f"dataset/results/proprietary/{openai_model}/{prompt_type}")

if not open_model_results_path.exists():
    open_model_results_path.mkdir(parents=True)
if not open_model_instructor_results_path.exists():
    open_model_instructor_results_path.mkdir(parents=True)
if not proprietary_results_path.exists():
    proprietary_results_path.mkdir(parents=True)


def get_response_from_openai(scrape_result, file_name):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_format = prompt_module.response_format
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} review articles in JSON format.",
            },
            {
                "role": "user",
                "content": "Extract the following information in JSON format: "
                + scrape_result,
            },
        ],
        max_tokens=16384,
        temperature=0.0,
        response_format=response_format,
    )    
    json_string = response.choices[0].message.content
    json_object = json.loads(json_string)
    with open(proprietary_results_path / f"{file_name}.json", "w") as f:
        json.dump(json_object, f)


def get_response_from_open_source_with_instructor(scrape_result, file_name):
    # client = OpenAI(base_url="http://localhost:8888/v1")
    # client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    client = OpenAI(base_url="http://Osprey2.csl.illinois.edu:8000/v1")
    client = instructor.from_openai(client)
    prompt_module = importlib.import_module(f"prompt.{prompt_type}")
    response_model = getattr(prompt_module, prompt_type.capitalize())
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} articles in JSON format.",
            },
            {"role": "user", "content": f"The article of the {prompt_type} is: " + scrape_result},
        ],
        max_tokens=16384,
        temperature=0.0,
        response_model=response_model,
    )
    json_object = response.json()
    with open(open_model_instructor_results_path / f"{file_name}.json", "w") as f:
        json.dump(json_object, f)

def process_file(file, openai_func, open_source_instructor_func):
    with open(file, "r") as f:
        scrape_result = f.read()

    file_name = file.stem

    # Run all three functions concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(openai_func, scrape_result, file_name),
            executor.submit(open_source_instructor_func, scrape_result, file_name)
        ]
        concurrent.futures.wait(futures)

# Main execution
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    file_processor = partial(process_file, 
                             openai_func=get_response_from_openai,
                             open_source_instructor_func=get_response_from_open_source_with_instructor)
    
    list(executor.map(file_processor, dataset_path.iterdir()))
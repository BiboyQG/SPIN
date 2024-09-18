from openai import OpenAI
import pathlib as pl
import importlib
import json
import os

dataset_path = pl.Path("dataset/article")
open_source_model = "Qwen/Qwen2-72B-Instruct-AWQ"
prompt_type = "car"
prompt_module = importlib.import_module(f"prompt.{prompt_type}")

for file in dataset_path.iterdir():
    with open(file, "r") as f:
        scrape_result = f.read()

    file_name = file.stem

    # OpenAI call here
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response_format = prompt_module.response_format
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    with open(f"dataset/results/proprietary/{file_name}.json", "w") as f:
        json.dump(json_object, f)

    # Open-source call here
    client = OpenAI(base_url="http://Osprey1.csl.illinois.edu:8000/v1")
    llm_prompt = prompt_module.llm_prompt
    query = llm_prompt.format(scrape_result)
    response = client.chat.completions.create(
        model=open_source_model,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at summarizing {prompt_type} review articles in JSON format.",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=26000,
        temperature=0.0,
    )
    json_string = response.choices[0].message.content
    json_object = json.loads(json_string)
    with open(f"dataset/results/open-source/{file_name}.json", "w") as f:
        json.dump(json_object, f)
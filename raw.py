from openai import OpenAI
from prompt.professor import Prof
import json
prompt_type = "prof"

with open("./dataset/article/prof/0.txt", "r") as file:
    scrape_result = file.read()

open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"

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
        extra_body={
            "guided_json": Prof.model_json_schema()
        },
    )
    response_text = response.choices[0].message.content
    json_data = json.loads(response_text)
    print(json_data)


get_response_from_open_source_with_extra_body(scrape_result)



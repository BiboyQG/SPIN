from openai import OpenAI
# from prompt.prof import response_format

prompt_type = "prof"

with open("./dataset/article/prof/0.txt", "r") as file:
    scrape_result = file.read()

open_source_model = "Qwen/Qwen2.5-72B-Instruct-AWQ"

response_format = {
    "properties": {
        "fullname": {"type": "string"},
        "title": {"type": "string"},
        "contact": {
            "type": "object",
            "properties": {
                "phone": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["phone", "email"],
            "additionalProperties": False,
        },
        "office": {"type": "string"},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["degree", "field", "institution", "year"],
                "additionalProperties": False,
            },
        },
        "biography": {"type": "string"},
        "professionalHighlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "string"},
                    "organization": {"type": "string"},
                    "yearStart": {"type": "integer"},
                    "yearEnd": {"type": ["integer", "null"]},
                },
                "required": [
                    "position",
                    "organization",
                    "yearStart",
                    "yearEnd",
                ],
                "additionalProperties": False,
            },
        },
        "researchStatement": {"type": "string"},
        "researchInterests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["area", "description"],
                "additionalProperties": False,
            },
        },
        "researchAreas": {"type": "array", "items": {"type": "string"}},
        "publications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "conference": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["title", "authors", "conference", "year"],
                "additionalProperties": False,
            },
        },
        "teachingHonors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "honor": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["honor", "year"],
                "additionalProperties": False,
            },
        },
        "researchHonors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "honor": {"type": "string"},
                    "organization": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["honor", "organization", "year"],
                "additionalProperties": False,
            },
        },
        "coursesTaught": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["code", "title"],
                "additionalProperties": False,
            },
        },
    },
}

def get_response_from_open_source_with_instructor(scrape_result):
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
            "response_format": {"type": "json_object"},
            "guided_json": response_format["properties"],
            "guided_decoding_backend": "outlines",
        },
    )

    print(response)


get_response_from_open_source_with_instructor(scrape_result)



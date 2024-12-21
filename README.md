# Knowledge LLM Agent: Hierarchical Assessment Framework for Language Models

## Overview

This repository presents a novel hierarchical framework for Language Model (LLM) assessment that mirrors human's information retrieval process. Our approach integrates Finite State Machines (FSM) to achieve structured outputs, enabling systematic assessment and improved database operations.

## Setup

> [!IMPORTANT]
>
> The requirement below is mandatory. And we've only tested our project on the following platform.

| Mandatory     | Recommended |
| ------------- | ----------- |
| Python        | 3.10        |
| CUDA          | 12.2        |
| torch         | 2.5.1       |
| transformers  | 4.45.1      |
| vLLM          | 0.6.4.post1 |

> [!TIP]
>
> Use `pip install -r requirement.txt` to install all the requirement if you want to create a new environment on your own or stick with existing environment.

### Quickstart

#### Repo Download

We first clone the whole project by git clone this repo:

```bash
git clone git@github.com:Forward-UIUC-2024F/banghao-chi-knowledge-agent.git && cd banghao-chi-knowledge-agent
```

#### Environment Setup

Then, it is necessary for us to setup a virtual environmrnt in order to run the project.

Currently, we don't provide docker image or dockerfile. So we recommend you to use `conda` to setup the environment.

> [!NOTE]
>
> You can rename `my_new_env` to any name you want.

```bash
conda env create -n my_new_env python=3.10
```

And then activate the environment:

```bash
conda activate my_new_env
```

Install the requirement by running:

```bash
pip install -r requirements.txt
```

You should also install docker desktop on your own machine, which is required for setting up the database. If you don't have it, you can download it from [here](https://www.docker.com/products/docker-desktop/).

Once you have the environment ready, you can setup the database by running:

```bash
docker compose up -d
```

After the database is ready, you can migrate the database by running:

```bash
make migrate-up
```

#### Launching Server

Then we need to setup server-side to provide the service to the clients. To launch our OpenAI-compatible server, simply:

```bash
CUDA_VISIBLE_DEVICES=... NCCL_P2P_DISABLE=1 vllm serve YOUR_MODEL_NAME --gpu-memory-utilization=0.95 --trust-remote-code
```

## Codebase Structure

banghao-chi-knowledge-agent/
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── dataset/              # Data storage and results
│   ├── article/          # Raw article data
│   │   ├── prof/         # Professor profile articles
│   │   ├── car/          # Car review articles
│   │   └── movie/        # Movie review articles
│   ├── results/          # Evaluation results
│   │   ├── gt/           # Ground truth data
│   │   ├── open-source/  # Open-source model results
│   │   └── proprietary/  # Proprietary model results
│   └── source/           # Source URLs
├── prompt/               # Prompt templates
│   ├── prof.py           # Professor data schema
│   ├── car.py            # Car review schema
│   └── movie.py          # Movie review schema
├── create_dataset.py     # Dataset creation utilities
├── eval.py               # Evaluation metrics and scripts
├── generate_results.py   # Result generation
├── interface.py          # Web interface
├── multi.py              # Knowledge extraction and database operations
├── raw.py                # Raw data processing
├── database.py           # Database operations
└── outlines.ipynb        # FSM explanation and example

## Functional Design (Usage)

This framework provides several key functionalities for extracting structured information from text using LLMs and FSM-guided outputs.

### Core Extraction Functions

* Extract structured information from text using open-source models (e.g. Qwen/Qwen2.5-72B-Instruct-AWQ):

```python
def get_response_from_open_source_with_extra_body():
    client = OpenAI()
    response = client.chat.completions.create(
        model=open_source_model_name,
        messages=[
            {
                "role": "system",
                "content": "...",
            },
            {
                "role": "user",
                "content": "...",
            },
        ],
        extra_body={"guided_json": your_pydantic_model.model_json_schema()},
    )
    return response.choices[0].message.content
```

### Evaluation Functions

* Compare extracted JSON against ground truth:

```python
def compare_json_objects(ground_truth: dict, test_object: dict):
  """
  Evaluates extraction accuracy against ground truth.
  Returns:
    OrderedDict with metrics:
      json_validity: bool
      key_similarity: float
      value_exactness: float
      numeric_similarity: float
      string_similarity: float
  """
```

## Demo Video

[To be added.]

## Issues and Future Work

Currently, we only have a pipeline for professor data extraction. We will add more pipelines for other data types in the future, so that the entire framework can handle:

* Dynamic schema creation
* Multi-source data extraction
* Database operations
* Information effectiveness evaluation
* ...

## Contributors

* [Banghao Chi](https://biboyqg.github.io/)
* Advisor: [Kevin Chang](https://siebelschool.illinois.edu/about/people/faculty/kcchang)

## Research Environment

This research was conducted at the National Center for Supercomputing Applications (NCSA) at the University of Illinois Urbana-Champaign.

## License

[License information will be added]

## Acknowledgments

Special thanks to [Kevin Chang](https://siebelschool.illinois.edu/about/people/faculty/kcchang) and [NCSA](https://ncsa.illinois.edu/) for providing the research infrastructure and support for this project.

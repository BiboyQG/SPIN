from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Architecture(BaseModel):
    parameters: int = Field(..., description="Number of parameters in billions")
    context_length: int
    attention_mechanism: str = Field(
        description="e.g., 'Grouped Query Attention', 'Multi Head Attention', 'Vanilla Attention'",
    )
    token_count: Optional[int] = Field(description="Training tokens count in billions")


class TrainingMetrics(BaseModel):
    compute_hours: float = Field(description="Total GPU/TPU hours used in training")
    hardware_type: str = Field(
        description="e.g., 'NVIDIA A100', 'NVIDIA H100', 'AMD MI250X', 'Google TPU v5p'"
    )


class Benchmark(BaseModel):
    name: str = Field(description="e.g., 'HumanEval', 'MMLU', 'MATH' etc.")
    score: float
    shot_count: Optional[int] = Field(
        ..., description="Number of shots used in evaluation"
    )


class Publication(BaseModel):
    title: str
    authors: List[str]
    venue: str
    year: int
    url: Optional[str]


class License(BaseModel):
    name: str = Field(
        description="e.g., 'Apache 2.0', 'MIT', 'CC BY-NC-SA 4.0', 'A custom commercial license, the Llama 3.3 Community License' etc."
    )


class LanguageModel(BaseModel):
    name: str = Field(
        ..., description="e.g., 'GPT-2', 'Qwen-2.5', 'Llama-3.1', 'Mixtral-Large' etc."
    )
    developer: str = Field(
        ...,
        description="Organization or team that developed the model, e.g., 'OpenAI', 'Google', 'Meta', 'Anthropic', 'DeepSeek', 'Qwen', 'Mistral' etc.",
    )
    supported_languages: List[str]
    release_date: datetime

    architecture: Architecture
    training_metrics: TrainingMetrics

    knowledge_cutoff: Optional[str]

    limitations: List[str]

    benchmarks: List[Benchmark]
    publication: Optional[Publication]

    license: License

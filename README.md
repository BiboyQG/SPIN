# LLM Knowledge Agent System

An advanced Python-based research agent that performs deep, iterative research on entities and extracts structured information using Large Language Models.

## Overview

This system combines iterative research methodology with entity information extraction to gather comprehensive, structured data about entities (professors, companies, students, etc.) from web sources. It uses a multi-action approach inspired by advanced research agents, performing searches, visiting URLs, reflecting on gaps, and extracting structured data.

## Key Features

- **Automatic Entity Type Detection**: Automatically detects what type of entity is being researched
- **Iterative Research Process**: Uses SEARCH, VISIT, REFLECT, EXTRACT, and EVALUATE actions
- **Schema-Driven Extraction**: Maps findings to predefined schemas using guided JSON with LLMs
- **Smart URL Management**: Ranks and filters URLs based on relevance and diversity
- **Knowledge Accumulation**: Builds knowledge through multiple sources
- **Progress Tracking**: Detailed logging and progress reporting
- **Quality Evaluation**: Assesses completeness, accuracy, and consistency

## Architecture

### Core Components

1. **ResearchAgent** (`core/research_agent.py`)

   - Main orchestrator that coordinates the research process
   - Manages the iterative loop and action execution

2. **URLManager** (`core/url_manager.py`)

   - Discovers, ranks, and filters URLs
   - Ensures diversity and relevance

3. **KnowledgeAccumulator** (`core/knowledge_accumulator.py`)

   - Stores and consolidates research findings
   - Maps knowledge to schema fields

4. **SearchEngine** (`core/search_engine.py`)

   - Integrates with search APIs (Brave, DuckDuckGo, etc.)
   - Generates targeted queries

5. **ActionPlanner** (`core/action_planner.py`)
   - Decides next actions based on current state
   - Balances exploration vs exploitation

### Action Types

- **SEARCH**: Web search for relevant information
- **VISIT**: Deep content extraction from URLs
- **REFLECT**: Analyze knowledge gaps and plan next steps
- **EXTRACT**: Structure information into schema format
- **EVALUATE**: Assess research quality and completeness

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd SPIN
```

2. Install dependencies:

```bash
pip install -r requirements_research.txt
```

3. Set up environment variables:

```bash
cp .envrc.example .envrc
# Edit .envrc with your API keys:
# - OPENAI_API_KEY or compatible LLM API key
# - BRAVE_SEARCH_API_KEY (for search functionality)
# - OPENAI_BASE_URL (if using custom endpoint)
```

## Usage

### Basic Usage

```python
from core.research_agent import ResearchAgent

# Initialize the agent
agent = ResearchAgent()

# Research an entity (auto-detect type)
results = agent.research_entity("Geoffrey Hinton professor")

# Research with explicit type
results = agent.research_entity("John Smith MIT", entity_type="professor")

# Access results
print(f"Entity Type: {results['entity_type']}")
print(f"Extracted Data: {results['extracted_data']}")
print(f"Completeness: {results['metadata']['completeness']}%")
```

### Command Line Usage

```bash
# Run test script
python test_research.py "Geoffrey Hinton professor"

# Results will be saved to: research_results_Geoffrey_Hinton_professor.json
```

### Configuration

Create a custom configuration:

```python
from core.config import ResearchConfig, SearchConfig, LLMConfig

config = ResearchConfig(
    max_steps=30,
    max_tokens_budget=50000,
    max_urls_per_step=3,
    search_config=SearchConfig(
        provider="brave",
        max_results_per_query=10
    ),
    llm_config=LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.0
    )
)

agent = ResearchAgent(config)
```

## Research Process Flow

1. **Entity Detection**

   - Search for the entity
   - Analyze top result to detect entity type
   - Load appropriate schema

2. **Iterative Research Loop**

   - Plan next action based on current state
   - Execute action (search, visit, reflect, etc.)
   - Update knowledge base
   - Evaluate progress
   - Repeat until complete or budget exhausted

3. **Final Extraction**
   - Consolidate all findings
   - Extract structured data using LLM with guided JSON
   - Validate against schema

## Output Format

The system returns a comprehensive result dictionary:

```json
{
  "success": true,
  "entity_type": "professor",
  "query": "Geoffrey Hinton professor",
  "extracted_data": {
    "name": "Geoffrey Hinton",
    "title": "Professor",
    "department": "Computer Science",
    "university": "University of Toronto",
    "email": "...",
    "research_interests": [...],
    // ... other fields
  },
  "metadata": {
    "duration_seconds": 45.2,
    "fields_filled": 12,
    "total_fields": 15,
    "completeness": 80.0,
    "confidence_score": 0.85,
    "urls_visited": 8,
    "knowledge_items": 23,
    "tokens_used": 15000
  },
  "research_trail": {
    "actions_taken": [...],
    "search_queries": [...],
    "visited_urls": [...],
    "knowledge_summary": {...}
  }
}
```

## Extending the System

### Adding New Entity Types

1. Create a new schema in `schemas/` directory:

```python
# schemas/company.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Company(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    headquarters: Optional[str] = None
    website: Optional[str] = None
    # ... more fields
```

2. The system will automatically detect and use the new schema

### Custom Action Executors

Create new action types by extending `ActionExecutor`:

```python
from core.actions.base import ActionExecutor

class CustomExecutor(ActionExecutor):
    def execute(self, action, context):
        # Implementation
        pass
```

## Performance Considerations

- **Token Budget**: Monitor token usage to stay within limits
- **Rate Limiting**: Built-in delays between API calls
- **URL Diversity**: Limits URLs per domain to ensure broad coverage
- **Caching**: Consider implementing caching for repeated searches

## Troubleshooting

- **No search results**: Check API keys and network connectivity
- **Schema detection fails**: Ensure entity has a matching schema or implement schema generation
- **Low completeness**: Increase max_steps or adjust search strategies
- **Token budget exceeded**: Reduce max_urls_per_step or content chunk sizes

## Limitations

- Requires reliable internet connection
- Dependent on search API availability
- LLM accuracy affects extraction quality
- Some entity types may require custom schemas

## Future Enhancements

- API interface for integration
- Parallel URL processing
- Advanced caching system
- Multi-language support
- Real-time progress streaming
- Custom entity type generation

## License

[Your License Here]

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

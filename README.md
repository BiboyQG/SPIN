# Knowledge LLM Agent: Hierarchical Assessment Framework for Language Models

This repository presents a novel hierarchical framework for Language Model (LLM) assessment that mirrors human cognitive processes in grading and evaluation. Our approach integrates Finite State Machines (FSM) to achieve structured outputs, enabling systematic assessment and improved database operations.

## Key Features

- **Hierarchical Assessment Framework**
  - Mirrors human cognitive grading processes
  - Sequential understanding and concept extraction
  - Sophisticated prompt engineering
  - Systematic feedback generation

- **FSM Integration for Structured Outputs**
  - Enhanced output consistency
  - Improved database operation performance
  - Structured data generation from LLM responses

- **Comprehensive Model Testing**
  - Tested across 41 different LLM variants
  - Parameter ranges from 2B to 110B
  - Evaluation of both proprietary and open-weight models
  - Custom metrics for accuracy assessment

## Technical Architecture

### Hierarchical Framework
Our assessment framework follows a structured approach:
1. Initial Understanding Phase
2. Concept Extraction Layer
3. Systematic Feedback Generation
4. Comprehensive Evaluation

### FSM Integration
The Finite State Machine integration enables:
- Controlled output generation
- Structured data formatting
- Consistent database operations

## Model Coverage

Our testing encompasses:
- Models ranging from 2B to 110B parameters
- Both proprietary and open-weight architectures
- Various model architectures and training approaches
- Custom evaluation metrics for stability and accuracy

## Getting Started

### Prerequisites
```bash
pip install vllm
```
This is required for server side inference.

### Installation
```bash
git clone https://github.com/BiboyQG/spin.git
cd spin
```

## Usage

```bash
python interface.py
```
This is the interface for the client side.

## Results

Our framework demonstrates:
- Consistent performance across different model sizes
- Improved output structure through FSM integration
- Reliable assessment metrics
- Stable performance across various LLM architectures

## Contributors

- [Banghao Chi](https://biboyqg.github.io/)
- Advisor: [Kevin Chang](https://siebelschool.illinois.edu/about/people/faculty/kcchang)

## Research Environment

This research was conducted at the National Center for Supercomputing Applications (NCSA) at the University of Illinois Urbana-Champaign.

## License

[License information will be added]

## Acknowledgments

Special thanks to [Kevin Chang](https://siebelschool.illinois.edu/about/people/faculty/kcchang) and [NCSA](https://ncsa.illinois.edu/) for providing the research infrastructure and support for this project.

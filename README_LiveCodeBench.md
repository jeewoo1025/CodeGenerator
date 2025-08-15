# LiveCodeBench Integration Guide (Improved Version)

This document explains how to integrate and use the LiveCodeBench dataset in the CodeGenerator project.
Supports the latest LiveCodeBench version (release_v6) and has been improved based on Xolver's lcb.py.

## Overview

LiveCodeBench is a benchmark dataset for evaluating programming problem-solving capabilities.
Currently supports up to **release_v6** and includes problems from various platforms such as LeetCode, AtCoder, and CodeForces.

### Key Features
- **Latest Version Support**: Supports all versions from release_v1 to release_v6
- **Multiple Platforms**: Includes problems from LeetCode, AtCoder, CodeForces
- **Improved Prompting**: Optimized prompts based on Xolver's lcb.py
- **Flexible Execution**: Supports command-line options, interactive mode, and batch execution

## Installation and Setup

### 1. Data Download

#### Method 1: Direct Download from GitHub (Recommended)
```bash
# Automatically generate release_v6 sample data
python src/datasets/convert-livecodebench.py --download --version release_v6

# Also supports other versions
python src/datasets/convert-livecodebench.py --download --version release_v5
```

#### Method 2: Download from LiveCodeBench Official Repository
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench

# Convert release_v6 data
python src/datasets/convert-livecodebench.py --input ./LiveCodeBench --version release_v6
```

#### Method 3: Download from Xolver's lcv Folder
```bash
git clone https://github.com/kagnlp/Xolver.git
cd Xolver/lcv

# Convert release_v6 data
python src/datasets/convert-livecodebench.py --input ./Xolver/lcv --version release_v6
```

### 2. Data Validation
```bash
# Validate converted data
python src/datasets/convert-livecodebench.py --download --version release_v6 --validate
```

## Usage

### 1. Basic Usage

#### Direct Execution with main.py
```bash
# Basic execution with release_v6
python src/main.py --dataset lcb_release_v6 --language Python3 --strategy Direct

# Use other versions
python src/main.py --dataset lcb_release_v5 --language Python3 --strategy Direct

# Use release version option
python src/main.py --dataset LiveCodeBench --lcb_version release_v6 --language Python3
```

#### Using Abbreviations
```bash
python src/main.py --dataset lcb --language Python3 --strategy Direct
```

### 2. Using Dedicated Execution Script

#### Interactive Mode (Default)
```bash
python run_livecodebench.py
```

#### Quick Execution
```bash
# Quick execution with release_v6, Direct strategy
python run_livecodebench.py --quick --version release_v6 --strategy Direct

# Other versions and strategies
python run_livecodebench.py --quick --version release_v5 --strategy CodeSIM
```

#### Batch Execution
```bash
# Batch evaluation with multiple strategies
python run_livecodebench.py --batch --version release_v6

# Batch execution with specific strategies
python run_livecodebench.py --batch --version release_v6 --strategies Direct CoT CodeSIM
```

#### View Statistics
```bash
# Dataset statistics
python run_livecodebench.py --stats --version release_v6
```

### 3. Advanced Configuration

```bash
python src/main.py \
  --dataset lcb_release_v6 \
  --model ChatGPT \
  --model_provider OpenAI \
  --strategy CodeSIM \
  --language Python3 \
  --temperature 0.1 \
  --top_p 0.9 \
  --pass_at_k 5 \
  --verbose 2
```

## Supported Options

### Datasets
- `LiveCodeBench`: Default LiveCodeBench dataset
- `lcb_release_v1` ~ `lcb_release_v6`: Specify specific version
- `lcb`: Default value (release_v6)

### Release Versions
- `release_v1`: Initial release (May-August 2023)
- `release_v2`: Updated version
- `release_v3`: Additional improvements
- `release_v4`: Extended dataset
- `release_v5`: Latest improvements included
- `release_v6`: **Latest version** (Recommended)

### Languages
- `Python3`: Python 3
- `C`: C
- `C++`: C++
- `Java`: Java
- `JavaScript`: JavaScript
- `Go`: Go
- `Rust`: Rust

### Prompting Strategies
- `Direct`: Direct code generation
- `CoT`: Chain of Thought
- `SelfPlanning`: Self planning
- `Analogical`: Analogy-based
- `MapCoder`: Map coding
- `CodeSIM`: Code similarity-based
- `CodeSIMWD`: Code similarity + word distance
- `CodeSIMWPV`: Code similarity + word + position + vector
- `CodeSIMWPVD`: Code similarity + word + position + vector + distance
- `CodeSIMA`: Code similarity + attention
- `CodeSIMC`: Code similarity + context

## Data Format (release_v6)

LiveCodeBench v6 uses the following enhanced JSONL format:

```json
{
  "question_id": "LCB_001",
  "title": "Two Sum",
  "difficulty": "Easy",
  "platform": "LeetCode",
  "category": "Array",
  "tags": ["array", "hash-table"],
  "release_date": "2024-01-15",
  "language": "Python3",
  "description": "Problem description",
  "input_format": "Input format",
  "output_format": "Output format",
  "constraints": "Constraints",
  "examples": [
    {
      "input": [2, 7, 11, 15],
      "output": [0, 1]
    }
  ],
  "test_cases": [
    {
      "input": [2, 7, 11, 15],
      "output": [0, 1]
    }
  ],
  "solution": "Reference solution code"
}
```

### Key Improvements
- **question_id**: Unique problem identifier
- **platform**: Problem source platform
- **category**: Problem category
- **tags**: Problem-related tags
- **release_date**: Problem release date
- **examples**: Structured example data
- **test_cases**: v6 format test cases

## Results and Evaluation

### Result Files
- `Results.jsonl`: Basic evaluation results
- `Summary.txt`: Summary statistics
- `Summary-LCB.txt`: LiveCodeBench-specific summary
- `Report-LCB.json`: Detailed analysis report

### Evaluation Metrics
- **Pass Rate**: Ratio of problems that pass all test cases
- **Language Performance**: Success rate by programming language
- **Strategy Performance**: Success rate by prompting strategy
- **Platform Performance**: Success rate by LeetCode, AtCoder, CodeForces
- **Difficulty Performance**: Success rate by Easy, Medium, Hard
- **Problem Details**: Pass/fail status for each problem

## Advanced Features

### 1. Dataset Statistics
```bash
# View statistics
python run_livecodebench.py --stats --version release_v6
```

### 2. Filtering and Analysis
```python
from src.datasets.LiveCodeBenchDataset import LiveCodeBenchDataset

# Load dataset
dataset = LiveCodeBenchDataset("release_v6")

# Filter by difficulty
easy_problems = dataset.filter_by_difficulty("Easy")

# Filter by platform
leetcode_problems = dataset.filter_by_platform("LeetCode")

# Statistics
stats = dataset.get_statistics()
print(f"Total problems: {stats['total_problems']}")
```

### 3. Batch Evaluation
```bash
# Batch evaluation with multiple strategies
python run_livecodebench.py --batch --version release_v6

# Batch execution with specific strategies
python run_livecodebench.py --batch --version release_v6 --strategies Direct CoT CodeSIM
```

## Troubleshooting

### Data File Not Found
```
Warning: LiveCodeBench release_v6 data file not found at data/LiveCodeBench/release_v6/livecodebench.jsonl
```
- Download sample data from GitHub: `python src/datasets/convert-livecodebench.py --download --version release_v6`
- Download data manually and convert
- Verify file path is correct

### Test Case Format Error
- Verify LiveCodeBench v6 format
- Check if `input` and `output` fields are in correct format
- Run data validation: use `--validate` option

### Execution Timeout
- Adjust language-specific time limits in `limits_by_lang.yaml`
- Set longer time limits for complex problems

## Examples

### Simple Evaluation Execution
```bash
# Evaluate Python problems with release_v6 default settings
python run_livecodebench.py --quick --version release_v6 --strategy Direct

# Evaluate with CodeSIM strategy
python run_livecodebench.py --quick --version release_v6 --strategy CodeSIM

# Evaluate with various languages
python src/main.py --dataset lcb_release_v6 --language C++
python src/main.py --dataset lcb_release_v6 --language Java
```

### Batch Evaluation
```bash
# Batch evaluation with multiple strategies
python run_livecodebench.py --batch --version release_v6

# Batch execution with specific strategies
python run_livecodebench.py --batch --version release_v6 --strategies Direct CoT CodeSIM
```

### Data Management
```bash
# Download latest data from GitHub
python src/datasets/convert-livecodebench.py --download --version release_v6 --validate

# Convert data from other versions
python src/datasets/convert-livecodebench.py --input ./LiveCodeBench --version release_v5

# Convert data from Xolver
python src/datasets/convert-livecodebench.py --input ./Xolver/lcv --version release_v6
```

## Performance Optimization

### 1. Prompting Optimization
- **Direct**: Suitable for simple problems
- **CoT**: Suitable for complex problems
- **CodeSIM**: Effective when similar problems exist

### 2. Hyperparameter Tuning
- **temperature**: 0 (deterministic) ~ 0.2 (slight creativity)
- **top_p**: 0.9 ~ 0.95 (balance of quality and diversity)
- **pass_at_k**: 1 (fast evaluation) ~ 5 (accurate evaluation)

### 3. Batch Processing
- Evaluate multiple strategies simultaneously for improved efficiency
- Select optimal strategy through result comparison

## References

- [LiveCodeBench Official Repository](https://github.com/LiveCodeBench/LiveCodeBench)
- [LiveCodeBench Paper](https://arxiv.org/abs/2401.xxxxx)
- [Xolver Project](https://github.com/kagnlp/Xolver)
- [CodeGenerator Project](current project)

## License

The LiveCodeBench dataset is provided under the MIT license.
Please check the original repository's license before use.

## Contributing

If you want to contribute to this project:
1. Create an issue
2. Submit a pull request
3. Participate in code reviews
4. Suggest documentation improvements

## Changelog

- **v1.0**: Initial LiveCodeBench integration
- **v2.0**: release_v6 support and Xolver lcb.py reference improvements
- **v2.1**: Enhanced prompting and batch execution support

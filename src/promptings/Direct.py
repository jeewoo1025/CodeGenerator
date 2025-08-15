from typing import List
import tiktoken
import os
from copy import deepcopy

from .Base import BaseStrategy
from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results


class DirectStrategy(BaseStrategy):
    def run_single_pass(self, data_row: dict):
        # Use the new method from BaseStrategy
        if self.is_livecodebench():
            # Enhanced prompt for LiveCodeBench
            content = f"""{self.data.get_prompt(data_row)}

## Instructions for LiveCodeBench Problem:
1. **Read the problem carefully**: Understand the problem description, input/output format, and constraints
2. **Identify the algorithm**: Choose the most efficient approach (greedy, DP, graph, etc.)
3. **Handle edge cases**: Consider boundary conditions and special cases
4. **Write clean code**: Ensure your solution is readable and well-structured
5. **Test your logic**: Verify with the provided examples

Generate {self.language} code to solve the above mentioned problem:"""
        else:
            # Standard prompt for other datasets
            content = f'{self.data.get_prompt(data_row)}\n\nGenerate {self.language} code to solve the above mentioned problem:'
        
        processed_input = [
            {
                "role": "user",
                "content": content,
            },
        ]
        return self.gpt_chat(processed_input=processed_input)

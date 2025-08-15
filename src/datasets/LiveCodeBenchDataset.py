from datasets.Dataset import Dataset
from utils.jsonl import read_jsonl
import json
import os
from typing import Dict, List, Any, Optional


class LiveCodeBenchDataset(Dataset):
    def __init__(self, release_version: str = "release_v6"):
        """
        LiveCodeBench dataset class
        
        Args:
            release_version: LiveCodeBench release version (default: release_v6)
        """
        self.release_version = release_version
        self.data_path = f"data/LiveCodeBench/{release_version}/livecodebench.jsonl"
        super().__init__(self.data_path)
        self.id_key = "question_id"  # LiveCodeBench v6 uses question_id
        
        # Supported release versions
        self.supported_versions = [
            "release_v1", "release_v2", "release_v3", 
            "release_v4", "release_v5", "release_v6"
        ]
        
        if release_version not in self.supported_versions:
            print(f"Warning: {release_version} is not in supported versions: {self.supported_versions}")
            print("Falling back to release_v6")

    def load(self):
        """Load LiveCodeBench data"""
        try:
            self.data = read_jsonl(self.data_path)
            print(f"Loaded {len(self.data)} problems from LiveCodeBench {self.release_version}")
        except FileNotFoundError:
            self.data = []
            print(f"Warning: LiveCodeBench data file not found at {self.data_path}")
            print("Please download and convert the data using:")
            print(f"  python src/datasets/convert-livecodebench.py {self.release_version}")
            print("Available versions:", ", ".join(self.supported_versions))

    def evaluate(self, item: dict, cur_imp: str, language: str):
        """Evaluate LiveCodeBench problem"""
        from evaluations.evalute import livecodebench_evaluate
        return livecodebench_evaluate(cur_imp, item, language)

    @staticmethod
    def get_prompt(item: Dict[str, Any]) -> str:
        """
        Generate LiveCodeBench problem prompt
        Simplified version to allow prompting strategies to differentiate
        """
        # Problem basic information
        question_id = item.get('question_id', item.get('id', 'N/A'))
        title = item.get('title', 'N/A')
        difficulty = item.get('difficulty', 'N/A')
        platform = item.get('platform', 'N/A')
        
        # Problem description (LiveCodeBench v6 format)
        description = item.get('description', '')
        if not description:
            description = item.get('question', 'N/A')
        
        # Input/output format
        input_format = item.get('input_format', '')
        output_format = item.get('output_format', '')
        
        # Constraints
        constraints = item.get('constraints', '')
        
        # Examples (LiveCodeBench v6 format)
        examples = item.get('examples', [])
        examples_text = ""
        if examples:
            if isinstance(examples, list):
                for i, example in enumerate(examples, 1):
                    if isinstance(example, dict):
                        input_ex = example.get('input', '')
                        output_ex = example.get('output', '')
                        examples_text += f"Example {i}:\nInput: {input_ex}\nOutput: {output_ex}\n\n"
                    else:
                        examples_text += f"Example {i}: {example}\n"
            else:
                examples_text = str(examples)
        
        # Generate simplified prompt (similar to XCodeDataset style)
        prompt = f"Problem Description:\n{description}"
        
        if input_format:
            prompt += f"\nInput Specification:\n{input_format}"
        
        if output_format:
            prompt += f"\nOutput Specification:\n{output_format}"
        
        if constraints:
            prompt += f"\nConstraints:\n{constraints}"
        
        if examples_text:
            prompt += f"\nSample Examples:\n{examples_text}"
        
        # Add platform-specific notes
        if platform.lower() in ['leetcode', 'codeforces', 'atcoder']:
            prompt += f"\n\n-------\nImportant Note: This is a {platform} problem. Follow the standard input/output format for {platform}."
        
        return prompt

    def get_test_cases(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return test cases (supports LiveCodeBench v6 format)"""
        test_cases = item.get('test_cases', [])
        
        # LiveCodeBench v6 format conversion
        if test_cases and isinstance(test_cases, list):
            converted_cases = []
            for tc in test_cases:
                if isinstance(tc, dict):
                    # v6 format: {"input": [...], "output": [...]}
                    if 'input' in tc and 'output' in tc:
                        input_data = tc['input']
                        output_data = tc['output']
                        
                        # Convert input list to string
                        if isinstance(input_data, list):
                            input_str = '\n'.join(str(x) for x in input_data)
                        else:
                            input_str = str(input_data)
                        
                        # Use first element if output is a list
                        if isinstance(output_data, list):
                            output_str = str(output_data[0]) if output_data else ''
                        else:
                            output_str = str(output_data)
                        
                        converted_cases.append({
                            'input': input_str,
                            'output': output_str
                        })
                    # Legacy format: {"input": "...", "output": "..."}
                    elif 'input' in tc and 'output' in tc:
                        converted_cases.append(tc)
            
            return converted_cases
        
        return []

    def get_solution(self, item: Dict[str, Any]) -> str:
        """Return reference solution"""
        solution = item.get('solution', '')
        if not solution:
            solution = item.get('reference_solution', '')
        return solution

    def get_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Return problem metadata"""
        return {
            'question_id': item.get('question_id', item.get('id', '')),
            'title': item.get('title', ''),
            'difficulty': item.get('difficulty', ''),
            'platform': item.get('platform', ''),
            'category': item.get('category', ''),
            'tags': item.get('tags', []),
            'release_date': item.get('release_date', ''),
            'language': item.get('language', 'Python3')
        }

    def filter_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Filter problems by difficulty"""
        if not difficulty:
            return self.data
        
        filtered = []
        for item in self.data:
            if item.get('difficulty', '').lower() == difficulty.lower():
                filtered.append(item)
        
        return filtered

    def filter_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        """Filter problems by platform"""
        if not platform:
            return self.data
        
        filtered = []
        for item in self.data:
            if item.get('platform', '').lower() == platform.lower():
                filtered.append(item)
        
        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            return {}
        
        stats = {
            'total_problems': len(self.data),
            'difficulty_distribution': {},
            'platform_distribution': {},
            'category_distribution': {}
        }
        
        for item in self.data:
            # Difficulty distribution
            difficulty = item.get('difficulty', 'Unknown')
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
            
            # Platform distribution
            platform = item.get('platform', 'Unknown')
            stats['platform_distribution'][platform] = stats['platform_distribution'].get(platform, 0) + 1
            
            # Category distribution
            category = item.get('category', 'Unknown')
            stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
        
        return stats

    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get problem by ID"""
        for item in self.data:
            if item.get('question_id', item.get('id', '')) == problem_id:
                return item
        return None

    def get_problems_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get problems by tag"""
        filtered = []
        for item in self.data:
            tags = item.get('tags', [])
            if isinstance(tags, list) and tag.lower() in [t.lower() for t in tags]:
                filtered.append(item)
        return filtered

    def get_random_problems(self, count: int) -> List[Dict[str, Any]]:
        """Get random problems for sampling"""
        import random
        if count >= len(self.data):
            return self.data.copy()
        return random.sample(self.data, count)

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and return issues"""
        issues = {
            'missing_fields': [],
            'invalid_test_cases': [],
            'empty_descriptions': []
        }
        
        for i, item in enumerate(self.data):
            # Check required fields
            required_fields = ['question_id', 'title', 'description', 'test_cases']
            for field in required_fields:
                if field not in item or not item[field]:
                    issues['missing_fields'].append({
                        'index': i,
                        'id': item.get('question_id', item.get('id', 'unknown')),
                        'missing_field': field
                    })
            
            # Check test cases
            test_cases = item.get('test_cases', [])
            if not test_cases or not isinstance(test_cases, list):
                issues['invalid_test_cases'].append({
                    'index': i,
                    'id': item.get('question_id', item.get('id', 'unknown')),
                    'issue': 'No test cases or invalid format'
                })
            
            # Check description
            description = item.get('description', '')
            if not description or len(description.strip()) < 10:
                issues['empty_descriptions'].append({
                    'index': i,
                    'id': item.get('question_id', item.get('id', 'unknown')),
                    'description_length': len(description)
                })
        
        return issues

    def get_problem_type(self, item: Dict[str, Any]) -> str:
        """Determine the type of problem based on description and tags"""
        description = item.get('description', '').lower()
        tags = [tag.lower() for tag in item.get('tags', [])]
        
        # Problem type detection based on keywords
        if any(word in description for word in ['array', 'list', 'sequence']):
            return 'array'
        elif any(word in description for word in ['string', 'text', 'character']):
            return 'string'
        elif any(word in description for word in ['graph', 'tree', 'node', 'edge']):
            return 'graph'
        elif any(word in description for word in ['dynamic programming', 'dp', 'memoization']):
            return 'dynamic_programming'
        elif any(word in description for word in ['greedy', 'optimization']):
            return 'greedy'
        elif any(word in description for word in ['binary search', 'search']):
            return 'binary_search'
        elif any(word in description for word in ['sorting', 'order']):
            return 'sorting'
        else:
            return 'general'
    
    def get_algorithm_suggestions(self, item: Dict[str, Any]) -> List[str]:
        """Suggest algorithms based on problem type and difficulty"""
        problem_type = self.get_problem_type(item)
        difficulty = item.get('difficulty', '').lower()
        
        suggestions = {
            'array': ['two pointers', 'sliding window', 'prefix sum', 'monotonic stack'],
            'string': ['two pointers', 'sliding window', 'KMP algorithm', 'trie'],
            'graph': ['BFS', 'DFS', 'Dijkstra', 'Floyd-Warshall', 'Union-Find'],
            'dynamic_programming': ['memoization', 'tabulation', 'state compression'],
            'greedy': ['sorting', 'priority queue', 'local optimization'],
            'binary_search': ['binary search on answer', 'parametric search'],
            'sorting': ['quick sort', 'merge sort', 'heap sort', 'counting sort']
        }
        
        base_algorithms = suggestions.get(problem_type, ['brute force', 'optimization'])
        
        # Add difficulty-based suggestions
        if difficulty in ['easy', 'medium']:
            base_algorithms.extend(['simulation', 'implementation'])
        elif difficulty in ['hard', 'expert']:
            base_algorithms.extend(['advanced data structures', 'complex algorithms'])
        
        return base_algorithms
    
    def get_edge_cases(self, item: Dict[str, Any]) -> List[str]:
        """Suggest potential edge cases based on problem type"""
        problem_type = self.get_problem_type(item)
        constraints = item.get('constraints', '').lower()
        
        edge_cases = {
            'array': ['empty array', 'single element', 'all same elements', 'maximum/minimum values'],
            'string': ['empty string', 'single character', 'all same characters', 'special characters'],
            'graph': ['empty graph', 'single node', 'disconnected components', 'cycles'],
            'dynamic_programming': ['base cases', 'boundary conditions', 'overflow'],
            'greedy': ['tie-breaking', 'local vs global optimal', 'constraint violations'],
            'binary_search': ['leftmost/rightmost occurrence', 'not found case', 'duplicate elements'],
            'sorting': ['duplicate elements', 'already sorted', 'reverse sorted']
        }
        
        base_edge_cases = edge_cases.get(problem_type, ['boundary conditions', 'invalid input'])
        
        # Add constraint-based edge cases
        if 'negative' in constraints:
            base_edge_cases.append('negative numbers')
        if 'zero' in constraints:
            base_edge_cases.append('zero values')
        if 'large' in constraints or '10^' in constraints:
            base_edge_cases.append('large numbers (overflow)')
        
        return base_edge_cases

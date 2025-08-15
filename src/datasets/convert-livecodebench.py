#!/usr/bin/env python3
"""
LiveCodeBench Data Conversion Script (Improved Version)

Converts data from LiveCodeBench official repository or Xolver's lcv folder
to a format usable by the system.
Improved based on Xolver's lcb.py.
"""

import json
import os
import sys
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class LiveCodeBenchConverter:
    """LiveCodeBench data converter"""
    
    def __init__(self):
        self.supported_versions = [
            "release_v1", "release_v2", "release_v3",
            "release_v4", "release_v5", "release_v6"
        ]
        
    def convert_livecodebench_v6_format(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LiveCodeBench v6 data to system format"""
        converted = {
            'question_id': input_data.get('question_id', input_data.get('id', '')),
            'title': input_data.get('title', ''),
            'difficulty': input_data.get('difficulty', 'Unknown'),
            'platform': input_data.get('platform', 'Unknown'),
            'category': input_data.get('category', ''),
            'tags': input_data.get('tags', []),
            'release_date': input_data.get('release_date', ''),
            'language': input_data.get('language', 'Python3'),
            'description': input_data.get('description', input_data.get('question', '')),
            'input_format': input_data.get('input_format', ''),
            'output_format': input_data.get('output_format', ''),
            'constraints': input_data.get('constraints', ''),
            'examples': input_data.get('examples', []),
            'test_cases': [],
            'solution': input_data.get('solution', input_data.get('reference_solution', ''))
        }
        
        # Test case conversion (v6 format)
        if 'test_cases' in input_data:
            test_cases = input_data['test_cases']
            if isinstance(test_cases, list):
                for tc in test_cases:
                    if isinstance(tc, dict):
                        # v6 format: {"input": [...], "output": [...]}
                        if 'input' in tc and 'output' in tc:
                            input_data = tc['input']
                            output_data = tc['output']
                            
                            # Keep input as list if it's already a list
                            if isinstance(input_data, list):
                                input_val = input_data
                            else:
                                input_val = [input_data]
                            
                            # Keep output as list if it's already a list
                            if isinstance(output_data, list):
                                output_val = output_data
                            else:
                                output_val = [output_data]
                            
                            converted['test_cases'].append({
                                'input': input_val,
                                'output': output_val
                            })
                        # Legacy format: {"input": "...", "output": "..."}
                        elif 'input' in tc and 'output' in tc:
                            converted['test_cases'].append(tc)
        
        # Generate default test case if none exist
        if not converted['test_cases']:
            converted['test_cases'] = [
                {'input': ['default_input'], 'output': ['default_output']}
            ]
        
        return converted

    def convert_from_livecodebench_repo(self, repo_path: str, output_path: str, version: str = "release_v6"):
        """Convert data from LiveCodeBench official repository"""
        print(f"Converting data from LiveCodeBench {version} repository: {repo_path}")
        
        # Find problem data files
        data_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.json') or file.endswith('.jsonl'):
                    # Version filtering
                    if version in root or version.replace('_', '') in root:
                        data_files.append(os.path.join(root, file))
        
        if not data_files:
            print(f"No data files found for version {version}")
            return False
        
        print(f"Found {len(data_files)} data files")
        
        converted_data = []
        
        for data_file in data_files:
            try:
                if data_file.endswith('.json'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                converted = self.convert_livecodebench_v6_format(item)
                                converted_data.append(converted)
                        else:
                            converted = self.convert_livecodebench_v6_format(data)
                            converted_data.append(converted)
                elif data_file.endswith('.jsonl'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                converted = self.convert_livecodebench_v6_format(item)
                                converted_data.append(converted)
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
                continue
        
        # Save converted data
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in converted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Conversion completed: {len(converted_data)} problems saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving converted data: {e}")
            return False

    def convert_from_xolver_lcv(self, lcv_path: str, output_path: str, version: str = "release_v6"):
        """Convert data from Xolver's lcv folder"""
        print(f"Converting data from Xolver's lcv folder for {version}: {lcv_path}")
        
        # Find problem data files
        data_files = []
        for root, dirs, files in os.walk(lcv_path):
            for file in files:
                if file.endswith('.json') or file.endswith('.jsonl'):
                    # Version filtering
                    if version in root or version.replace('_', '') in root:
                        data_files.append(os.path.join(root, file))
        
        if not data_files:
            print(f"No data files found for version {version}")
            return False
        
        print(f"Found {len(data_files)} data files")
        
        converted_data = []
        
        for data_file in data_files:
            try:
                if data_file.endswith('.json'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                converted = self.convert_livecodebench_v6_format(item)
                                converted_data.append(converted)
                        else:
                            converted = self.convert_livecodebench_v6_format(data)
                            converted_data.append(converted)
                elif data_file.endswith('.jsonl'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                converted = self.convert_livecodebench_v6_format(item)
                                converted_data.append(converted)
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
                continue
        
        # Save converted data
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in converted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Conversion completed: {len(converted_data)} problems saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving converted data: {e}")
            return False

    def download_livecodebench_data(self, version: str = "release_v6", output_dir: str = "data/LiveCodeBench"):
        """Download LiveCodeBench data from official repository"""
        print(f"Downloading LiveCodeBench {version} data...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download URLs for different versions
        download_urls = {
            "release_v1": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v1/livecodebench.jsonl",
            "release_v2": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v2/livecodebench.jsonl",
            "release_v3": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v3/livecodebench.jsonl",
            "release_v4": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v4/livecodebench.jsonl",
            "release_v5": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v5/livecodebench.jsonl",
            "release_v6": "https://raw.githubusercontent.com/livecodebench/livecodebench/main/data/release_v6/livecodebench.jsonl"
        }
        
        if version not in download_urls:
            print(f"Version {version} not supported for download")
            return False
        
        try:
            url = download_urls[version]
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw data
            raw_output_path = os.path.join(output_dir, f"{version}_raw.jsonl")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Raw data downloaded to {raw_output_path}")
            
            # Convert and save
            output_path = os.path.join(output_dir, version, "livecodebench.jsonl")
            success = self.convert_from_livecodebench_repo(
                os.path.dirname(raw_output_path), 
                output_path, 
                version
            )
            
            if success:
                # Clean up raw file
                os.remove(raw_output_path)
                print(f"Data successfully converted and saved to {output_path}")
                return True
            else:
                print("Conversion failed, raw data kept for manual processing")
                return False
                
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

    def validate_converted_data(self, data_path: str) -> Dict[str, Any]:
        """Validate converted data format"""
        print(f"Validating converted data: {data_path}")
        
        if not os.path.exists(data_path):
            return {"valid": False, "error": "File not found"}
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            validation_results = {
                "valid": True,
                "total_problems": len(data),
                "missing_fields": [],
                "invalid_test_cases": [],
                "empty_descriptions": []
            }
            
            for i, item in enumerate(data):
                # Check required fields
                required_fields = ['question_id', 'title', 'description', 'test_cases']
                for field in required_fields:
                    if field not in item or not item[field]:
                        validation_results['missing_fields'].append({
                            'index': i,
                            'id': item.get('question_id', item.get('id', 'unknown')),
                            'missing_field': field
                        })
                
                # Check test cases
                test_cases = item.get('test_cases', [])
                if not test_cases or not isinstance(test_cases, list):
                    validation_results['invalid_test_cases'].append({
                        'index': i,
                        'id': item.get('question_id', item.get('id', 'unknown')),
                        'issue': 'No test cases or invalid format'
                    })
                
                # Check description
                description = item.get('description', '')
                if not description or len(description.strip()) < 10:
                    validation_results['empty_descriptions'].append({
                        'index': i,
                        'id': item.get('question_id', item.get('id', 'unknown')),
                        'description_length': len(description)
                    })
            
            # Determine overall validity
            if (validation_results['missing_fields'] or 
                validation_results['invalid_test_cases'] or 
                validation_results['empty_descriptions']):
                validation_results['valid'] = False
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench Data Converter")
    parser.add_argument("--version", "-v", default="release_v6", 
                       choices=["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6"],
                       help="LiveCodeBench version to convert")
    parser.add_argument("--input", "-i", help="Input directory or file path")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--download", "-d", action="store_true", help="Download data from official repository")
    parser.add_argument("--validate", action="store_true", help="Validate converted data")
    
    args = parser.parse_args()
    
    # Set output path
    if not args.output:
        args.output = f"data/LiveCodeBench/{args.version}/livecodebench.jsonl"
    
    converter = LiveCodeBenchConverter()
    
    if args.download:
        # Download and convert
        success = converter.download_livecodebench_data(args.version)
        if success:
            print("Download and conversion completed successfully")
        else:
            print("Download and conversion failed")
            sys.exit(1)
    elif args.input:
        # Convert from local source
        if os.path.isdir(args.input):
            success = converter.convert_from_livecodebench_repo(args.input, args.output, args.version)
        else:
            print("Input must be a directory")
            sys.exit(1)
        
        if success:
            print("Conversion completed successfully")
        else:
            print("Conversion failed")
            sys.exit(1)
    else:
        print("Please specify --input or --download")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        validation_results = converter.validate_converted_data(args.output)
        if validation_results['valid']:
            print("Data validation passed")
        else:
            print("Data validation failed:")
            print(f"Missing fields: {len(validation_results['missing_fields'])}")
            print(f"Invalid test cases: {len(validation_results['invalid_test_cases'])}")
            print(f"Empty descriptions: {len(validation_results['empty_descriptions'])}")
            if 'error' in validation_results:
                print(f"Error: {validation_results['error']}")


if __name__ == "__main__":
    main()

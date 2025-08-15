# download_lcb.py
from datasets import load_dataset
import argparse

"""
python download_lcb.py --version release_v6 --output livecodebench_v6.jsonl
"""

def download_livecodebench(version="release_v6", output_file=None):
    if output_file is None:
        output_file = f"livecodebench_{version}.jsonl"
    
    print(f"Downloading LiveCodeBench {version}...")
    dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version)
    
    print(f"Saving to {output_file}...")
    dataset['train'].to_json(output_file)
    
    print(f"Done! Saved {len(dataset['train'])} problems to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="release_v6", help="Version to download")
    parser.add_argument("--output", help="Output file name")
    
    args = parser.parse_args()
    download_livecodebench(args.version, args.output)
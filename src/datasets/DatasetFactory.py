from datasets.Dataset import Dataset
from datasets.MBPPDataset import MBPPDataset
from datasets.APPSDataset import APPSDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.LiveCodeBenchDataset import LiveCodeBenchDataset


class DatasetFactory:
    @staticmethod
    def get_dataset_class(dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name == "apps":
            return APPSDataset
        elif dataset_name == "mbpp":
            return MBPPDataset
        elif dataset_name == "xcode":
            return XCodeDataset
        elif dataset_name == "xcodeeval":
            return XCodeDataset
        elif dataset_name == "humaneval":
            return HumanDataset
        elif dataset_name == "human":
            return HumanDataset
        elif dataset_name == "cc":
            return CodeContestDataset
        elif dataset_name == "livecodebench":
            return LiveCodeBenchDataset
        elif dataset_name == "lcb":
            return LiveCodeBenchDataset
        elif dataset_name.startswith("lcb_"):
            # Support lcb_release_v6 format
            version = dataset_name.replace("lcb_", "")
            return lambda release_version=version: LiveCodeBenchDataset(release_version=release_version)
        else:
            raise Exception(f"Unknown dataset name {dataset_name}")

    @staticmethod
    def create_dataset(dataset_name, **kwargs):
        """Create dataset instance with parameters"""
        dataset_class = DatasetFactory.get_dataset_class(dataset_name)
        
        # Handle LiveCodeBench with release version
        if dataset_name.lower() in ["livecodebench", "lcb"] or dataset_name.startswith("lcb_"):
            if dataset_name.startswith("lcb_"):
                version = dataset_name.replace("lcb_", "")
            else:
                version = kwargs.get('release_version', 'release_v6')
            return dataset_class(release_version=version)
        else:
            return dataset_class(**kwargs)

import warnings
from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Parses the requirements.txt file and returns a list of requirements,
    excluding comments, empty lines, and the editable installation directive (`-e .`).
    """
    try:
        with open(file_path, encoding="utf-8") as file_obj:
            # Read and clean lines, skipping comments and empty lines
            requirements = [
                req.strip()
                for req in file_obj
                if req.strip()
                and not req.strip().startswith("#")
                and req.strip() != HYPHEN_E_DOT
            ]
        return requirements
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        warnings.warn(
            f"Warning: {file_path} not found. No dependencies were installed from this file."
        )
        return []


setup(
    name="mlproject",
    version="0.0.1",
    author="kjpou1",
    author_email="wasssssuuuuupppp@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

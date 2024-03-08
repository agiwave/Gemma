import io
import os
import setuptools
from typing import List

ROOT_DIR = os.path.dirname(__file__)

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

setuptools.setup(
    name="Gemma",
    version="0.1",
    author="Bing Wu",
    author_email="85222460@qq.com",
    license="Apache 2.0",
    description=("A toolkit for AI"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=("benchmarks", "docs", "examples")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
)

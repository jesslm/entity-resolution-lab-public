#!/usr/bin/env python3
"""
Setup script for the Entity Resolution Demo package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = "../README.md"  # README is in the parent directory
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    else:
        return "Entity Resolution Demo Package"

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

# Read tutorial requirements
def read_tutorial_requirements():
    tutorial_requirements = [
        "jupyter>=1.0.0",
        "jupyterlab>=3.0.0",
        "nbformat>=5.0.0",
        "matplotlib>=3.0.0",
        "pandas>=1.0.0",
        "ipywidgets>=7.0.0",
        "graphviz>=0.20.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ]
    return tutorial_requirements

setup(
    name="entity-resolution-demo",
    version="1.0.0",
    author="Entity Resolution Demo Team",
    author_email="team@entityresolution.demo",
    description="A comprehensive educational prototype for entity resolution",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/entity-resolution-demo/entity-resolution-demo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "tutorial": read_tutorial_requirements(),
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "entity-resolution-demo=entity_resolution_demo.pipeline_runner.run_pipeline:main",
        ],
    },
)

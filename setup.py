#!/usr/bin/env python3
"""
BMW Agents - A Framework For Task Automation Through Multi-Agent Collaboration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bmw-agents",
    version="0.1.0",
    author="BMW Group Information Technology Research Center",
    author_email="marcin.ziolkowski@bmwgroup.com",
    description="A framework for task automation through multi-agent collaboration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmwgroup/bmw-agents",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bmw_agents": [
            "configs/prompt_templates/*.txt",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "ollama>=0.1.0",
        "tenacity>=8.0.0",
        "aiohttp>=3.8.0",
        "jinja2>=3.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
) 
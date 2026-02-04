"""
AECP - Agent Embedding Communication Protocol

Setup script for PyPI distribution.
"""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Agent Embedding Communication Protocol"

# Read version from package
version = "1.0.0"

setup(
    name="aecp",
    version=version,
    author="AECP Contributors",
    author_email="aecp@example.com",
    description="Agent Embedding Communication Protocol - Enable AI agents to communicate with 97% semantic fidelity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aecp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/aecp/issues",
        "Source": "https://github.com/yourusername/aecp",
        "Documentation": "https://aecp.dev/docs",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "voyage": ["voyageai>=0.1.0"],
        "cohere": ["cohere>=4.0.0"],
        "huggingface": ["sentence-transformers>=2.0.0"],
        "all": [
            "openai>=1.0.0",
            "voyageai>=0.1.0",
            "cohere>=4.0.0",
            "sentence-transformers>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.990",
            "flake8>=4.0.0",
        ],
    },
    keywords=[
        "ai", "agents", "embeddings", "semantic", "communication",
        "multi-agent", "llm", "machine-learning", "nlp", "vector",
    ],
    include_package_data=True,
    zip_safe=False,
)

"""
Setup script for SafeDeepAgent
World's Most Comprehensive Secure Agentic AI Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="safedeepagent",
    version="0.1.0",
    author="Oluwafemi Idiakhoa",
    author_email="Oluwafemidiakhoa@gmail.com",
    description="World's Most Comprehensive Secure Agentic AI Framework - 12 Security Foundations, Continual Learning, Multi-LLM Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oluwafemidiakhoa/Deepagent",
    project_urls={
        "Homepage": "https://github.com/oluwafemidiakhoa/Deepagent",
        "Documentation": "https://github.com/oluwafemidiakhoa/Deepagent#readme",
        "Repository": "https://github.com/oluwafemidiakhoa/Deepagent",
        "Issues": "https://github.com/oluwafemidiakhoa/Deepagent/issues",
        "Changelog": "https://github.com/oluwafemidiakhoa/Deepagent/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "benchmarks", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "tenacity>=8.2.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.1.0",
    ],
    extras_require={
        # LLM Provider Options
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
        ],
        "llm-all": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "ollama>=0.1.0",
            "litellm>=1.0.0",  # Unified interface for 100+ LLMs (DeepSeek, Qwen, etc.)
        ],
        "llm-local": [
            "ollama>=0.1.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ],
        # Embeddings & Vector
        "embeddings": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
        ],
        "embeddings-gpu": [
            "sentence-transformers>=2.2.0",
            "faiss-gpu>=1.7.0",
        ],
        # Vector Databases
        "vector": [
            "chromadb>=0.4.0",
            "qdrant-client>=1.6.0",
        ],
        # Traditional Databases
        "database": [
            "psycopg2-binary>=2.9.0",
            "redis>=5.0.0",
        ],
        # Observability
        "observability": [
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
        # Development
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        # Complete installation
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "ollama>=0.1.0",
            "litellm>=1.0.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "chromadb>=0.4.0",
            "qdrant-client>=1.6.0",
            "psycopg2-binary>=2.9.0",
            "redis>=5.0.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
        # Minimal production setup
        "minimal": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
        ],
    },
    keywords=[
        "ai", "agents", "agentic-ai", "llm", "security",
        "machine-learning", "deep-learning", "openai", "anthropic",
        "deepseek", "qwen", "continual-learning", "seal",
        "autonomous-agents", "agent-security", "prompt-injection",
        "memory-firewall", "deception-detection",
    ],
)
